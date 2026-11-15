# Free Model Aggregator

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/GeMseN-Test/free-model-aggregator/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/GeMseN-Test/free-model-aggregator/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

Entry for the [make.fun bounty "The best free AI model aggregator"](https://make.fun/bounty/free-model-aggregator-01) (`free-model-aggregator-01`, $500).

One server, Python 3.13 stdlib only (`main.py`, no dependencies). It reads the
brief from `MAKEFUN_PROMPT_FILE`, probes each provider's catalog route at
startup, registers with the env at `$MAKEFUN_ENV_URL`, and serves the 20
requests one at a time — each answered by the highest-rated **free** model
that meets the request's requirements, speaking that provider's wire
(OpenAI / Anthropic / Gemini), with failover to the next-best model when a
provider refuses (402/429).

## Run

```bash
python3 main.py            # serve a trial (needs MAKEFUN_PROMPT_FILE + MAKEFUN_ENV_URL)
python3 main.py --list     # print the catalog JSON and exit 0
```

## Design

See [STRATEGY.md](STRATEGY.md).

## Local scores (practice trials, sealed scorer)

| trial | score | quality | served | wire | failover | catalog | upstream calls |
|------:|------:|--------:|-------:|-----:|---------:|--------:|---------------:|
| 0 | 99 | 99 | 100 | 100 | 100 | 100 | 24 |
| 1 | 99 | 99 | 100 | 100 | 100 | 100 | 30 |
| 2 | 99 | 99 | 100 | 100 | 100 | 100 | 27 |

Quality 99 is the ceiling on these trials: the sealed per-trial budgets and
failure schedules force at least a few requests onto the second-best (1470)
model. The state export is byte-identical across re-runs (replay-safe).

## License

MIT — see [LICENSE](LICENSE).
