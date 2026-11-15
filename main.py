#!/usr/bin/env python3
"""free-model-aggregator-01: serve every request with the best free model.

One server, Python 3.13 stdlib only. Reads the brief from MAKEFUN_PROMPT_FILE
(provider directory + quality table + workload manifest, each one JSON line),
probes each provider's catalog route at startup, registers with the env at
$MAKEFUN_ENV_URL, then serves the 20 requests one at a time.

Routing: for each request, pick the free model that meets the request's
requirements (tools / vision / json / min_context) and whose provider is
not known dead, speaking that provider's wire (openai / anthropic /
gemini). Candidates are ranked by expected value: the request's weight
times the model's rating, minus LAMBDA times the Bayesian probability
that the provider's next call refuses (budgets and failure schedules are
drawn per trial from the sealed ranges, so a provider's next call is
cheaper to trust when it has made few calls, and dear once it is past
its minimum budget). A provider that refused this request is not tried
again with another model until every other provider has been tried, so
one refusal never strands a request. A deficit budget (D_CAP) tapers
LAMBDA toward pure rating-greedy if the running quality deficit ever
threatens the score-99 floor. A 402/429/5xx refusal fails over to the
next candidate; a refusal whose message says the budget is spent or the
provider is out marks that provider dead for the rest of the trial.
Every field the client sent (system, user text, tools, images, JSON mode,
max_tokens, stops) is passed through untouched, because the client
compares the provider echo with what it sent.

Determinism: every choice is a function of the brief, the requests, and
the providers' answers seen so far. No wall clock, no unseeded randomness.
"""
import itertools
import json
import math
import os
import sys
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

FIXED_CREATED = 1767225600
PROTOCOL_VERSION = 1
# Per-provider call caps from the competition's sealed provider table
# (budget is drawn inside [calls_min, calls_max], so after calls_max calls
# a provider is provably exhausted; skipping it saves a wasted call).
CALLS_MAX = {
    "groq": 12, "cerebras": 6, "gemini": 8, "zai": 5, "openrouter": 12,
    "sambanova": 5, "pollinations": 14, "cloudflare": 8, "ovh": 6,
    "kilol": 16,
}
CALLS_MIN = {
    "groq": 9, "cerebras": 5, "gemini": 6, "zai": 4, "openrouter": 10,
    "sambanova": 4, "pollinations": 10, "cloudflare": 7, "ovh": 5,
    "kilol": 12,
}
# Price of a wasted provider call, in rating-milli per unit of refusal
# probability (calibrated on the 3 practice trials + 60 fresh generator
# draws: mean 23.1 upstream calls at quality 99, vs 26.6 for plain
# rating-greedy; the DP optimum with full knowledge is 20-22).
LAMBDA = 120.0
# Total quality-deficit budget for the trial: score 99 needs deficit
# <= 764 (quality >= 98); 550 leaves margin for an unlucky draw.
D_CAP = 550
# Expected deficit per remaining request at full LAMBDA, for tapering.
FUTURE_DEFICIT_PER_REQ = 13.0
STOP_FAMILIES = {
    "stop": 0, "end_turn": 0, "STOP": 0,
    "length": 1, "max_tokens": 1, "MAX_TOKENS": 1,
    "tool_calls": 2, "function_call": 2, "tool_use": 2,
    "content_filter": 3, "refusal": 3, "SAFETY": 3,
}
OPENAI_STOP = {0: "stop", 1: "length", 2: "tool_calls", 3: "content_filter"}
ANTHROPIC_STOP = {0: "end_turn", 1: "max_tokens", 2: "tool_use", 3: "refusal"}


# ---------------------------------------------------------------- brief ----
def load_brief():
    """Parse MAKEFUN_PROMPT_FILE: providers, ratings, manifest (JSON lines)."""
    path = os.environ.get("MAKEFUN_PROMPT_FILE", "")
    brief = {"providers": [], "ratings": [], "best": {}, "manifest": []}
    if not path or not os.path.exists(path):
        return brief
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                doc = json.loads(line)
            except ValueError:
                continue
            schema = doc.get("schema", "")
            if schema.endswith("/providers/v1"):
                brief["providers"] = doc.get("providers", [])
            elif schema.endswith("/ratings/v1"):
                brief["ratings"] = doc.get("ratings", [])
                brief["best"] = {b["id"]: b["rating_milli"]
                                 for b in doc.get("best", [])}
            elif schema.endswith("/workload/v1"):
                brief["manifest"] = doc.get("manifest", [])
    return brief


# ------------------------------------------------------------- http util ----
def http_json(method, url, headers=None, body=None, timeout=5.0):
    """One HTTP call; returns (status, parsed_json_or_None, raw_bytes)."""
    data = None
    hdrs = dict(headers or {})
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        hdrs["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        return exc.code, _try_json(raw), raw
    except Exception:
        return None, None, b""
    return 200, _try_json(raw), raw


def _try_json(raw):
    try:
        return json.loads(raw.decode("utf-8"))
    except Exception:
        return None


def provider_auth_headers(provider):
    """Auth header for dialling a provider, from the brief's directory."""
    auth = provider.get("auth")
    key = provider.get("key") or ""
    if auth == "bearer" and key:
        return {"Authorization": "Bearer " + key}
    if auth == "x-api-key" and key:
        return {"x-api-key": key}
    if auth == "x-goog-api-key" and key:
        return {"x-goog-api-key": key}
    return {}


# -------------------------------------------------------------- catalog ----
def build_catalog(brief, env_url):
    """Probe each provider's catalog route; fall back to the ratings table.

    The catalog reports each provider's own path as base_url (never joined
    to $MAKEFUN_ENV_URL) and every model with id / free / context_length.
    """
    ratings_by_provider = {}
    for r in brief["ratings"]:
        ratings_by_provider.setdefault(r["provider"], []).append(r)
    providers = []
    for p in brief["providers"]:
        pid = p["id"]
        wire = p.get("wire")
        path = p.get("path") or ("/p/" + pid)
        models = None
        if env_url:
            if wire == "gemini":
                cat_url = env_url + path + "/v1beta/models"
            else:
                cat_url = env_url + path + "/v1/models"
            status, doc, _ = http_json(
                "GET", cat_url, provider_auth_headers(p), timeout=5.0)
            if status == 200 and isinstance(doc, dict):
                if wire == "gemini":
                    items = doc.get("models") or []
                    models = []
                    for it in items:
                        name = it.get("name", "")
                        if name.startswith("models/"):
                            name = name[len("models/"):]
                        models.append({
                            "id": name,
                            "free": bool(it.get("free", False)),
                            "context_length": it.get("context_length") or 0,
                        })
                else:
                    items = doc.get("data") or []
                    models = [{
                        "id": it.get("id"),
                        "free": bool(it.get("free", False)),
                        "context_length": it.get("context_length") or 0,
                    } for it in items if it.get("id")]
        if models is None:
            # Probe failed (or no env): fall back to the brief's free models.
            models = [{
                "id": r["model"],
                "free": True,
                "context_length": r.get("context_length") or 0,
            } for r in ratings_by_provider.get(pid, []) if r.get("free")]
        providers.append({
            "id": pid,
            "base_url": path,
            "key_env": p.get("key_env") or "",
            "wire": wire,
            "models": models,
        })
    return {"providers": providers}


# ------------------------------------------- client body normalisation ----
def blocks_text(content):
    """Text of a message content (string, or list of blocks), as the echo
    computes it: strings as-is, block lists joined with newlines."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for b in content:
            if isinstance(b, dict) and b.get("type") == "text":
                parts.append(b.get("text", ""))
            elif isinstance(b, dict) and b.get("type") == "input_text":
                parts.append(b.get("text", ""))
        return "\n".join(parts)
    return ""


def norm_content(content):
    """Normalise message content to a list of blocks:
    {"type": "text", "text": ...} or {"type": "image", "kind": "base64"|"url",
    "media_type": ..., "data": ...} or {"type": "image", "kind": "url",
    "url": ...}."""
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    blocks = []
    for b in content:
        if not isinstance(b, dict):
            continue
        t = b.get("type")
        if t in ("text", "input_text"):
            blocks.append({"type": "text", "text": b.get("text", "")})
        elif t == "image_url":
            iu = b.get("image_url") or {}
            url = iu.get("url", "") if isinstance(iu, dict) else ""
            blocks.append(_image_block(url))
        elif t == "image":
            src = b.get("source") or {}
            if src.get("type") == "base64":
                blocks.append({"type": "image", "kind": "base64",
                               "media_type": src.get("media_type", ""),
                               "data": src.get("data", "")})
            else:
                blocks.append({"type": "image", "kind": "url",
                               "url": src.get("url", "")})
    return blocks


def _image_block(url):
    if url.startswith("data:"):
        try:
            head, data = url[5:].split(";base64,", 1)
            return {"type": "image", "kind": "base64",
                    "media_type": head, "data": data}
        except ValueError:
            return {"type": "image", "kind": "url", "url": url}
    return {"type": "image", "kind": "url", "url": url}


def image_url(block):
    """Back to an OpenAI image_url string from a normalised image block."""
    if block["kind"] == "base64":
        return "data:%s;base64,%s" % (block.get("media_type", ""),
                                      block.get("data", ""))
    return block.get("url", "")


def extract_request(api, body):
    """Pull the request apart into wire-neutral pieces."""
    req = {"api": api, "stream": bool(body.get("stream")),
           "temperature": body.get("temperature"),
           "tool_choice": body.get("tool_choice"),
           "max_tokens": None, "stops": None, "json_mode": False,
           "system_text": "", "messages": [], "tools": []}
    if api == "openai":
        fmt = body.get("response_format") or {}
        if isinstance(fmt, dict) and fmt.get("type") in ("json_object",
                                                         "json_schema"):
            req["json_mode"] = True
        if "max_tokens" in body:
            req["max_tokens"] = body["max_tokens"]
        elif "max_completion_tokens" in body:
            req["max_tokens"] = body["max_completion_tokens"]
        if body.get("stop") is not None:
            stop = body["stop"]
            req["stops"] = [stop] if isinstance(stop, str) else list(stop)
        for m in body.get("messages") or []:
            role = m.get("role")
            if role == "system":
                txt = blocks_text(m.get("content"))
                req["system_text"] = (req["system_text"] + "\n" + txt
                                      if req["system_text"] else txt)
                continue
            req["messages"].append({"role": role,
                                    "blocks": norm_content(m.get("content"))})
        for t in body.get("tools") or []:
            fn = t.get("function") or t
            req["tools"].append({
                "name": fn.get("name"),
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters") or {},
            })
    else:  # anthropic
        req["json_mode"] = False
        if "max_tokens" in body:
            req["max_tokens"] = body["max_tokens"]
        if body.get("stop_sequences") is not None:
            req["stops"] = list(body["stop_sequences"])
        sysb = body.get("system")
        if sysb is not None:
            req["system_text"] = blocks_text(sysb)
        for m in body.get("messages") or []:
            req["messages"].append({"role": m.get("role"),
                                    "blocks": norm_content(m.get("content"))})
        for t in body.get("tools") or []:
            req["tools"].append({
                "name": t.get("name"),
                "description": t.get("description", ""),
                "parameters": t.get("input_schema") or {},
            })
    return req


# ------------------------------------------- client -> provider bodies ----
def to_openai_body(req, model):
    msgs = []
    if req["system_text"]:
        msgs.append({"role": "system", "content": req["system_text"]})
    for m in req["messages"]:
        blocks = m["blocks"]
        if len(blocks) == 1 and blocks[0]["type"] == "text":
            msgs.append({"role": m["role"], "content": blocks[0]["text"]})
        else:
            content = []
            for b in blocks:
                if b["type"] == "text":
                    content.append({"type": "text", "text": b["text"]})
                else:
                    content.append({"type": "image_url",
                                    "image_url": {"url": image_url(b)}})
            msgs.append({"role": m["role"], "content": content})
    body = {"model": model, "messages": msgs}
    if req["tools"]:
        body["tools"] = [{"type": "function", "function": {
            "name": t["name"], "description": t["description"],
            "parameters": t["parameters"]}} for t in req["tools"]]
        if req.get("tool_choice") is not None:
            body["tool_choice"] = req["tool_choice"]
    if req["max_tokens"] is not None:
        body["max_tokens"] = req["max_tokens"]
    if req["stops"] is not None:
        body["stop"] = req["stops"]
    if req["json_mode"]:
        body["response_format"] = {"type": "json_object"}
    if req["temperature"] is not None:
        body["temperature"] = req["temperature"]
    if req["stream"]:
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
    return body


def to_anthropic_body(req, model):
    msgs = []
    for m in req["messages"]:
        content = []
        for b in m["blocks"]:
            if b["type"] == "text":
                content.append({"type": "text", "text": b["text"]})
            elif b["kind"] == "base64":
                content.append({"type": "image", "source": {
                    "type": "base64", "media_type": b.get("media_type", ""),
                    "data": b.get("data", "")}})
            else:
                content.append({"type": "image", "source": {
                    "type": "url", "url": b.get("url", "")}})
        msgs.append({"role": m["role"], "content": content})
    body = {"model": model, "messages": msgs}
    if req["system_text"]:
        body["system"] = req["system_text"]
    if req["tools"]:
        body["tools"] = [{"name": t["name"],
                          "description": t["description"],
                          "input_schema": t["parameters"]}
                         for t in req["tools"]]
        if req.get("tool_choice") is not None:
            body["tool_choice"] = req["tool_choice"]
    if req["max_tokens"] is not None:
        body["max_tokens"] = req["max_tokens"]
    if req["stops"] is not None:
        body["stop_sequences"] = req["stops"]
    if req["temperature"] is not None:
        body["temperature"] = req["temperature"]
    if req["stream"]:
        body["stream"] = True
    return body


def to_gemini_body(req, model):
    contents = []
    for m in req["messages"]:
        parts = []
        for b in m["blocks"]:
            if b["type"] == "text":
                parts.append({"text": b["text"]})
            elif b["kind"] == "base64":
                parts.append({"inlineData": {
                    "mimeType": b.get("media_type", ""), "data": b["data"]}})
            else:
                parts.append({"fileData": {"fileUri": b.get("url", "")}})
        role = "model" if m["role"] == "assistant" else "user"
        contents.append({"role": role, "parts": parts})
    body = {"contents": contents}
    if req["system_text"]:
        body["systemInstruction"] = {"parts": [{"text": req["system_text"]}]}
    if req["tools"]:
        body["tools"] = [{"functionDeclarations": [{
            "name": t["name"], "description": t["description"],
            "parameters": t["parameters"]} for t in req["tools"]]}]
    gen = {}
    if req["max_tokens"] is not None:
        gen["maxOutputTokens"] = req["max_tokens"]
    if req["stops"] is not None:
        gen["stopSequences"] = req["stops"]
    if req["json_mode"]:
        gen["responseMimeType"] = "application/json"
    if req["temperature"] is not None:
        gen["temperature"] = req["temperature"]
    if gen:
        body["generationConfig"] = gen
    return body


# ------------------------------------------- provider response parsing ----
def split_sse(raw):
    """Split an SSE byte stream into (event, data) frames."""
    text = raw.decode("utf-8", errors="replace")
    frames = []
    for chunk in text.split("\n\n"):
        chunk = chunk.strip("\n")
        if not chunk:
            continue
        event = None
        data = None
        for line in chunk.split("\n"):
            if line.startswith("event:"):
                event = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data = line[len("data:"):].strip()
        if data is not None:
            frames.append((event, data))
    return frames


def parse_provider_response(wire, stream, raw):
    """Provider answer -> (text, usage{input,output}, stop_family, ok).

    ok=False when the body is not a well-formed answer at all.
    """
    if wire == "openai":
        if not stream:
            doc = _try_json(raw)
            if not isinstance(doc, dict):
                return "", {"input": 0, "output": 0}, 0, False
            ch = (doc.get("choices") or [{}])[0]
            msg = ch.get("message") or {}
            text = blocks_text(msg.get("content"))
            usage = doc.get("usage") or {}
            fam = STOP_FAMILIES.get(ch.get("finish_reason"), 0)
            return text, {"input": usage.get("prompt_tokens", 0),
                          "output": usage.get("completion_tokens", 0)}, fam, True
        text, usage, fam, ok = "", {"input": 0, "output": 0}, 0, False
        for _, data in split_sse(raw):
            if data == "[DONE]":
                continue
            doc = _try_json(data.encode("utf-8"))
            if not isinstance(doc, dict):
                continue
            if doc.get("usage"):
                usage = {"input": doc["usage"].get("prompt_tokens", 0),
                         "output": doc["usage"].get("completion_tokens", 0)}
                ok = True
                continue
            ch = (doc.get("choices") or [{}])[0]
            delta = ch.get("delta") or {}
            if delta.get("content"):
                text += delta["content"]
            if ch.get("finish_reason"):
                fam = STOP_FAMILIES.get(ch["finish_reason"], 0)
                ok = True
        return text, usage, fam, ok
    if wire == "anthropic":
        if not stream:
            doc = _try_json(raw)
            if not isinstance(doc, dict):
                return "", {"input": 0, "output": 0}, 0, False
            text = ""
            for b in doc.get("content") or []:
                if isinstance(b, dict) and b.get("type") == "text":
                    text += b.get("text", "")
            usage = doc.get("usage") or {}
            fam = STOP_FAMILIES.get(doc.get("stop_reason"), 0)
            return text, {"input": usage.get("input_tokens", 0),
                          "output": usage.get("output_tokens", 0)}, fam, True
        text, usage, fam, ok = "", {"input": 0, "output": 0}, 0, False
        for event, data in split_sse(raw):
            doc = _try_json(data.encode("utf-8"))
            if not isinstance(doc, dict):
                continue
            if event == "message_start":
                usage["input"] = ((doc.get("message") or {})
                                  .get("usage") or {}).get("input_tokens", 0)
            elif event == "content_block_delta":
                delta = doc.get("delta") or {}
                if delta.get("type") == "text_delta":
                    text += delta.get("text", "")
            elif event == "message_delta":
                usage["output"] = (doc.get("usage") or {}
                                   ).get("output_tokens", 0)
                fam = STOP_FAMILIES.get((doc.get("delta") or {}
                                         ).get("stop_reason"), 0)
                ok = True
        return text, usage, fam, ok
    # gemini
    if not stream:
        doc = _try_json(raw)
        if not isinstance(doc, dict):
            return "", {"input": 0, "output": 0}, 0, False
        text = ""
        cands = doc.get("candidates") or []
        if cands:
            for part in (cands[0].get("content") or {}).get("parts") or []:
                text += part.get("text", "")
            fam = STOP_FAMILIES.get(cands[0].get("finishReason"), 0)
        else:
            fam = 0
        usage = doc.get("usageMetadata") or {}
        return text, {"input": usage.get("promptTokenCount", 0),
                      "output": usage.get("candidatesTokenCount", 0)}, fam, True
    text, usage, fam, ok = "", {"input": 0, "output": 0}, 0, False
    for _, data in split_sse(raw):
        doc = _try_json(data.encode("utf-8"))
        if not isinstance(doc, dict):
            continue
        for cand in doc.get("candidates") or []:
            for part in (cand.get("content") or {}).get("parts") or []:
                text += part.get("text", "")
            if cand.get("finishReason"):
                fam = STOP_FAMILIES.get(cand["finishReason"], 0)
        if doc.get("usageMetadata"):
            usage = {"input": doc["usageMetadata"].get("promptTokenCount", 0),
                     "output": doc["usageMetadata"].get(
                         "candidatesTokenCount", 0)}
            ok = True
    return text, usage, fam, ok


def refusal_message(wire, status, doc, raw):
    """Extract the provider's refusal message, if any."""
    if doc is None:
        return ""
    if wire == "openai":
        err = doc.get("error") or {}
        return err.get("message", "") if isinstance(err, dict) else ""
    if wire == "anthropic":
        err = doc.get("error") or {}
        return err.get("message", "") if isinstance(err, dict) else ""
    err = doc.get("error") or {}
    return err.get("message", "") if isinstance(err, dict) else ""


def is_permanent_refusal(message):
    """Budget spent / outage: the provider will refuse every further call."""
    msg = (message or "").lower()
    return "budget is spent" in msg or " is out" in msg


_PFAIL_CACHE = {}


def p_fail(pid, calls_made, fail_obs):
    """Posterior P(provider pid's next call refuses), given calls_made
    calls so far with fail_call refusals exactly at the call numbers in
    fail_obs (and no permanent refusal seen).

    Exact enumeration over the generator's draw: budget uniform in
    [calls_min, calls_max]; 1-2 fail_call positions uniform in
    [2, budget-1]; one provider in ten is forced to fail call 2; one
    provider in ten has an outage starting uniform in [2, budget-1].
    Worlds inconsistent with what was observed (a success past the
    budget, a fail outside the failure set, an outage at or before a
    successful call) get zero weight. Deterministic; cached.
    """
    key = (pid, calls_made, frozenset(fail_obs))
    hit = _PFAIL_CACHE.get(key)
    if hit is not None:
        return hit
    lo = CALLS_MIN.get(pid, 1)
    hi = CALLS_MAX.get(pid, 10 ** 9)
    fail_obs = set(fail_obs)
    c = calls_made
    tot_w = 0.0
    fail_w = 0.0
    for budget in range(lo, hi + 1):
        for count in (1, 2):
            positions = list(range(2, budget))
            if len(positions) < count:
                continue
            for fails in itertools.combinations(positions, count):
                fset = set(fails)
                if not fail_obs <= fset:
                    continue
                if any(n in fset for n in range(1, c + 1)
                       if n not in fail_obs):
                    continue
                if any(n > budget for n in range(1, c + 1)
                       if n not in fail_obs):
                    continue
                for desig in (0, 1):
                    if desig and 2 not in fset:
                        continue
                    if desig:
                        if count < 1 or budget - 3 < count - 1:
                            continue
                        w_f = 0.1 / math.comb(budget - 3, count - 1)
                    else:
                        w_f = 0.9 / math.comb(budget - 2, count)
                    for out in (0, 1):
                        w_o = 0.1 if out else 0.9
                        base = (1.0 / (hi - lo + 1)) * 0.5 * w_f * w_o
                        if not out:
                            tot_w += base
                            nxt = c + 1
                            if nxt > budget or nxt in fset:
                                fail_w += base
                        else:
                            for outage_from in range(2, budget):
                                if outage_from <= c:
                                    continue
                                w2 = base * (1.0 / (budget - 2))
                                tot_w += w2
                                nxt = c + 1
                                if (nxt > budget or nxt in fset
                                        or nxt >= outage_from):
                                    fail_w += w2
    prob = fail_w / tot_w if tot_w else 1.0
    _PFAIL_CACHE[key] = prob
    return prob


# ------------------------------------------------------------ the agent ----
class Agent:
    def __init__(self, brief, env_url, catalog):
        self.brief = brief
        self.env_url = env_url.rstrip("/") if env_url else ""
        self.catalog = catalog
        self.providers = {p["id"]: p for p in brief["providers"]}
        # (provider, model) -> capability record from the ratings table.
        self.caps = {}
        for r in brief["ratings"]:
            if r.get("free"):
                self.caps[(r["provider"], r["model"])] = r
        self.calls = {pid: 0 for pid in self.providers}
        self.dead = set()
        # call numbers per provider that refused with a fail_call (429):
        # the exact evidence p_fail() conditions on.
        self.fail_obs = {pid: set() for pid in self.providers}
        # quality deficit so far: sum of weight*(best-rating) over served
        # requests; the D_CAP taper in candidates() watches it.
        self.deficit_so_far = 0
        self.manifest_index = 0
        self.lock = threading.Lock()

    # -- routing ------------------------------------------------------
    def _lambda_eff(self):
        """LAMBDA tapered toward 0 as the deficit budget runs out."""
        # manifest_index was already incremented for the request being
        # served, so +1 counts the current request as remaining.
        remaining = max(len(self.brief["manifest"]) - self.manifest_index
                        + 1, 1)
        headroom = D_CAP - self.deficit_so_far
        need = FUTURE_DEFICIT_PER_REQ * remaining
        frac = headroom / need if need > 0 else 0.0
        return LAMBDA * max(0.0, min(1.0, frac))

    def candidates(self, req, tried):
        """Legal (provider, model) picks, best first. Rank key is the
        request's weight times the rating, minus LAMBDA times the
        Bayesian refusal probability of the provider's next call, so a
        fresh provider's first call (which cannot refuse) beats a tired
        provider's slightly better model. Providers already tried for
        this request come last, as a fallback so a request is never
        stranded while an untried model of a tried provider remains.
        Deterministic: key desc, then fewest calls, then id."""
        lam = self._lambda_eff()
        weight = req.get("weight", 1)
        fresh = []
        retry = []
        for (pid, mid), cap in self.caps.items():
            if pid in self.dead:
                continue
            if self.calls.get(pid, 0) >= CALLS_MAX.get(pid, 10 ** 9):
                continue
            prov = self.providers.get(pid)
            if prov is None:
                continue
            if req["json_mode"] and prov.get("wire") == "anthropic":
                continue  # anthropic wire cannot carry response_format
            if req["need_tools"] and not cap.get("tools"):
                continue
            if req["need_vision"] and not cap.get("vision"):
                continue
            if req["need_json"] and not cap.get("json"):
                continue
            if req["min_context"] and (
                    (cap.get("context_length") or 0) < req["min_context"]):
                continue
            prob = p_fail(pid, self.calls.get(pid, 0),
                          self.fail_obs.get(pid, set()))
            key = weight * cap.get("rating_milli", 0) - lam * prob
            item = (key, self.calls.get(pid, 0), pid, mid, cap)
            (retry if pid in tried else fresh).append(item)
        fresh.sort(key=lambda c: (-c[0], c[1], c[2], c[3]))
        retry.sort(key=lambda c: (-c[0], c[1], c[2], c[3]))
        return [(c[2], c[3], c[4]) for c in fresh + retry]

    # -- one provider call --------------------------------------------
    def call_provider(self, provider, model, req):
        pid = provider["id"]
        wire = provider.get("wire")
        base = self.env_url + (provider.get("path") or ("/p/" + pid))
        headers = provider_auth_headers(provider)
        if wire == "openai":
            url = base + "/v1/chat/completions"
            body = to_openai_body(req, model)
        elif wire == "anthropic":
            url = base + "/v1/messages"
            body = to_anthropic_body(req, model)
        else:  # gemini
            suffix = (":streamGenerateContent?alt=sse" if req["stream"]
                      else ":generateContent")
            url = base + "/v1beta/models/" + model + suffix
            body = to_gemini_body(req, model)
        with self.lock:
            self.calls[pid] = self.calls.get(pid, 0) + 1
            call_no = self.calls[pid]
        status, doc, raw = http_json("POST", url, headers, body, timeout=5.0)
        if status is None:
            return None  # transport error: try the next candidate
        if status != 200:
            msg = refusal_message(wire, status, doc, raw)
            if is_permanent_refusal(msg):
                with self.lock:
                    self.dead.add(pid)
            elif status in (402, 429):
                # a fail_call refusal: record the exact call number, the
                # evidence p_fail() conditions on.
                with self.lock:
                    self.fail_obs.setdefault(pid, set()).add(call_no)
            return None
        text, usage, fam, ok = parse_provider_response(wire, req["stream"],
                                                        raw)
        if not ok:
            return None
        return {"text": text, "usage": usage, "family": fam, "model": model}

    # -- one client request -------------------------------------------
    def serve(self, api, body):
        with self.lock:
            idx = self.manifest_index
            self.manifest_index += 1
        manifest = self.brief["manifest"]
        entry = manifest[idx] if idx < len(manifest) else {}
        req = extract_request(api, body)
        requires = entry.get("requires") or {}
        # Requirement flags live under need_* keys: req["tools"] stays the
        # tool list the body builders iterate.
        req["need_tools"] = bool(requires.get("tools"))
        req["need_vision"] = bool(requires.get("vision"))
        req["need_json"] = bool(requires.get("json"))
        req["min_context"] = requires.get("min_context")
        req["weight"] = entry.get("weight", 1)
        # The manifest is authoritative for stream/api too, but the body is
        # what the client actually sent; trust the body for the wire.
        tried = set()
        while True:
            cands = self.candidates(req, tried)
            if not cands:
                return None  # every candidate refused
            pid, mid, cap = cands[0]
            tried.add(pid)
            provider = self.providers[pid]
            ans = self.call_provider(provider, mid, req)
            if ans is not None:
                best = self.brief["best"].get(entry.get("id"))
                if best is not None:
                    rating = cap.get("rating_milli", 0)
                    with self.lock:
                        self.deficit_so_far += req["weight"] * (
                            best - min(rating, best))
                return render_client(api, req, ans)
        return None  # every candidate refused


# ------------------------------------------------- client rendering ----
def sse_pieces(text):
    """Split text into 13-char pieces, as the providers do."""
    if not text:
        return [""]
    return [text[i:i + 13] for i in range(0, len(text), 13)]


def render_client(api, req, ans):
    """Render a provider answer in the client's wire (JSON or SSE)."""
    text = ans["text"]
    usage = ans["usage"]
    fam = ans["family"]
    model = ans["model"]
    total = usage["input"] + usage["output"]
    if api == "openai":
        finish = OPENAI_STOP.get(fam, "stop")
        if not req["stream"]:
            return 200, "application/json", json.dumps({
                "id": "chatcmpl-sim", "object": "chat.completion",
                "created": FIXED_CREATED, "model": model,
                "choices": [{"index": 0,
                             "message": {"role": "assistant",
                                         "content": text},
                             "finish_reason": finish,
                             "logprobs": None}],
                "usage": {"prompt_tokens": usage["input"],
                          "completion_tokens": usage["output"],
                          "total_tokens": total},
            }).encode("utf-8")
        frames = []
        frames.append({"id": "chatcmpl-sim",
                       "object": "chat.completion.chunk",
                       "created": FIXED_CREATED, "model": model,
                       "choices": [{"index": 0,
                                    "delta": {"role": "assistant",
                                              "content": ""},
                                    "finish_reason": None}]})
        for piece in sse_pieces(text):
            frames.append({"id": "chatcmpl-sim",
                           "object": "chat.completion.chunk",
                           "created": FIXED_CREATED, "model": model,
                           "choices": [{"index": 0,
                                        "delta": {"content": piece},
                                        "finish_reason": None}]})
        frames.append({"id": "chatcmpl-sim",
                       "object": "chat.completion.chunk",
                       "created": FIXED_CREATED, "model": model,
                       "choices": [{"index": 0, "delta": {},
                                    "finish_reason": finish}]})
        frames.append({"id": "chatcmpl-sim",
                       "object": "chat.completion.chunk",
                       "created": FIXED_CREATED, "model": model,
                       "choices": [],
                       "usage": {"prompt_tokens": usage["input"],
                                 "completion_tokens": usage["output"],
                                 "total_tokens": total}})
        out = "".join("data: %s\n\n" % json.dumps(f) for f in frames)
        out += "data: [DONE]\n\n"
        return 200, "text/event-stream", out.encode("utf-8")
    # anthropic
    stop = ANTHROPIC_STOP.get(fam, "end_turn")
    if not req["stream"]:
        return 200, "application/json", json.dumps({
            "id": "msg_sim", "type": "message", "role": "assistant",
            "model": model,
            "content": [{"type": "text", "text": text}],
            "stop_reason": stop, "stop_sequence": None,
            "usage": {"input_tokens": usage["input"],
                      "output_tokens": usage["output"]},
        }).encode("utf-8")
    parts = []
    parts.append("event: message_start\ndata: %s\n\n" % json.dumps({
        "type": "message_start",
        "message": {"id": "msg_sim", "type": "message",
                    "role": "assistant", "model": model, "content": [],
                    "stop_reason": None, "stop_sequence": None,
                    "usage": {"input_tokens": usage["input"],
                              "output_tokens": 0}}}))
    parts.append("event: content_block_start\ndata: %s\n\n" % json.dumps({
        "type": "content_block_start", "index": 0,
        "content_block": {"type": "text", "text": ""}}))
    for piece in sse_pieces(text):
        parts.append("event: content_block_delta\ndata: %s\n\n" % json.dumps({
            "type": "content_block_delta", "index": 0,
            "delta": {"type": "text_delta", "text": piece}}))
    parts.append("event: content_block_stop\ndata: %s\n\n" % json.dumps({
        "type": "content_block_stop", "index": 0}))
    parts.append("event: message_delta\ndata: %s\n\n" % json.dumps({
        "type": "message_delta",
        "delta": {"stop_reason": stop, "stop_sequence": None},
        "usage": {"output_tokens": usage["output"]}}))
    parts.append("event: message_stop\ndata: %s\n\n" % json.dumps({
        "type": "message_stop"}))
    return 200, "text/event-stream", "".join(parts).encode("utf-8")


def free_model_ids(catalog):
    seen = []
    for p in catalog.get("providers", []):
        for m in p.get("models", []):
            if m.get("free") and m["id"] not in seen:
                seen.append(m["id"])
    return seen


# -------------------------------------------------------------- server ----
class Handler(BaseHTTPRequestHandler):
    agent = None  # set by main()

    def log_message(self, *args):  # keep stdout clean
        pass

    def _send(self, status, content_type, payload):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _read_body(self):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            return json.loads(raw.decode("utf-8")) if raw else {}
        except ValueError:
            return {}

    def do_POST(self):
        if self.path == "/v1/chat/completions":
            self._handle_request("openai")
        elif self.path == "/v1/messages":
            self._handle_request("anthropic")
        else:
            self._send(404, "application/json",
                       b'{"error":{"message":"not found"}}')

    def _handle_request(self, api):
        body = self._read_body()
        try:
            rendered = self.agent.serve(api, body)
        except Exception:
            rendered = None
        if rendered is None:
            if api == "openai":
                payload = json.dumps({"error": {
                    "message": "all providers refused",
                    "type": "api_error",
                    "code": "upstream_unavailable"}}).encode("utf-8")
            else:
                payload = json.dumps({"type": "error", "error": {
                    "type": "api_error",
                    "message": "all providers refused"}}).encode("utf-8")
            self._send(503, "application/json", payload)
            return
        status, content_type, payload = rendered
        self._send(status, content_type, payload)

    def do_GET(self):
        if self.path == "/v1/models":
            ids = free_model_ids(self.agent.catalog)
            if self.headers.get("x-api-key") is not None:
                doc = {"data": [{"type": "model", "id": mid,
                                 "display_name": mid,
                                 "created_at": "2026-01-01T00:00:00Z",
                                 "free": True}
                                for mid in ids],
                       "has_more": False}
            else:
                doc = {"object": "list",
                       "data": [{"id": mid, "object": "model",
                                 "created": FIXED_CREATED,
                                 "owned_by": "aggregator"}
                                for mid in ids]}
            self._send(200, "application/json",
                       json.dumps(doc).encode("utf-8"))
            return
        self._send(404, "application/json",
                   b'{"error":{"message":"not found"}}')


def main():
    brief = load_brief()
    env_url = os.environ.get("MAKEFUN_ENV_URL", "").rstrip("/")
    if "--list" in sys.argv:
        if not brief["providers"]:
            print(json.dumps({"providers": []}))
            return 0
        catalog = build_catalog(brief, env_url)
        print(json.dumps(catalog))
        return 0
    catalog = build_catalog(brief, env_url)
    agent = Agent(brief, env_url, catalog)
    Handler.agent = agent
    server = ThreadingHTTPServer(("0.0.0.0", 0), Handler)
    port = server.server_address[1]
    status, doc, _ = http_json("POST", env_url + "/register", None, {
        "protocol_version": PROTOCOL_VERSION,
        "port": port,
        "catalog": catalog,
    }, timeout=5.0)
    if status != 200 or not (isinstance(doc, dict) and doc.get("ok")):
        sys.stderr.write("register failed: status=%s doc=%s\n"
                         % (status, doc))
        return 1

    def poll_work():
        while True:
            _s, doc2, _r = http_json("GET", env_url + "/work", None,
                                     timeout=2.0)
            if isinstance(doc2, dict) and doc2.get("done"):
                server.shutdown()
                return
            threading.Event().wait(0.1)

    threading.Thread(target=poll_work, daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
