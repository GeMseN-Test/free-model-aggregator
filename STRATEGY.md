# Strategy

## Goal

Serve all 20 requests validly, each with the highest-rated legal free model,
in as few provider calls as possible. Score = 60% quality + 15% served +
10% wire + 10% failover + 5% catalog; ties break on fewer provider calls.

## Routing

Greedy best-first, per request, from the brief's ratings table:

1. Legal candidates = free models meeting the request's requirements
   (tools / vision / json / min_context). A JSON-mode OpenAI request is
   never routed to an anthropic-wire provider (that wire cannot carry
   `response_format`, so the echo's `json_mode` would fail).
2. Sort by rating desc, then fewest calls made to that provider (spreads
   load across equally-rated providers), then provider/model id (deterministic).
3. Try in order until one answers. A 402/429/5xx refusal fails over to the
   next candidate. A refusal whose message says the budget is spent or the
   provider is out marks that provider dead for the rest of the trial.
4. A provider is also skipped once its call count reaches its sealed
   `calls_max` (budgets are drawn inside `[calls_min, calls_max]`, so past
   that point every further call is provably over budget).

The workload's weight-2 requests come first in manifest order, so greedy
naturally spends the best models where the weight is.

## Wire translation

Every field the client sent is passed through untouched — the client
compares the provider echo with what it sent, so a dropped or rewritten
field loses the request:

- system, user text, tools (names, in order), `max_tokens`, stops, images,
  JSON mode.
- Images keep their exact bytes/URL across wires (data URL ↔ Anthropic
  base64 block ↔ Gemini inlineData; http URL ↔ Anthropic url block ↔
  Gemini fileData).
- Provider answers are buffered, parsed into (text, usage, stop family),
  then re-rendered in the client's wire: OpenAI JSON/SSE (with a final
  usage chunk) or Anthropic JSON/SSE (message_start → content_block_start
  → deltas → content_block_stop → message_delta → message_stop).
- Stop reasons map by family: stop/end_turn/STOP → `stop` / `end_turn`,
  max_tokens → `length` / `max_tokens`, tool_use → `tool_calls` /
  `tool_use`, refusal → `content_filter` / `refusal`.

## Catalog

At startup the server probes each provider's catalog route
(`GET /v1/models`, Gemini `GET /v1beta/models`) with that provider's auth,
and registers the probed list (free + paid, each with `free` and
`context_length`). If a probe fails it falls back to the brief's ratings
table. `python3 main.py --list` prints exactly the catalog that gets
registered. The server's own `/v1/models` lists exactly the free model ids
in both auth shapes (Bearer → OpenAI list, x-api-key → Anthropic list).

## Determinism

Every choice is a function of the brief, the requests, and the providers'
answers seen so far. No wall clock, no unseeded randomness. The trial
state export is byte-identical across re-runs (verified on practice
trial-0), so the final replay check passes.

## What I would improve with more time

- Budget-aware planning: the greedy order is near-optimal, but a lookahead
  that reserves the 1470-tier providers for the requests where openrouter
  is most likely to fail could shave the last few quality points on
  unlucky trials (trial-1 r19: four providers refused in a row).
- A second-model retry on the same provider is sometimes a wasted call
  when a fresh provider offers the same rating; ranking the retry below a
  fresh provider at equal rating would save ~1 call per bad-luck trial.
