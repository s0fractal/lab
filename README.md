# lab

Експерименти. Кожен експеримент лежить в окремому каталозі: стартер (джерело та його транскрипція) плюс goal, за яким експеримент проводиться до вердикту.

| Каталог | Що це | Стан |
|---|---|---|
| [`phs-soliton/`](phs-soliton/) | Порт-гамільтонові та солітонні рекурентні резервуари проти GRU: [стартер](phs-soliton/starter/port-hamiltonian-soliton-reservoirs.md), [goal](phs-soliton/GOAL.md) | PREREG v1 (тег `prereg-v1`), 160 підтверджувальних прогонів, вердикт `INCONCLUSIVE`: [REPORT](phs-soliton/REPORT.md), [RESULTS](phs-soliton/RESULTS.md) |
| [`fringes/`](fringes/) | Пісочниця, не експеримент: post hoc розвідка смуг точності Soliton-PHS (пам'ять прецесує з кутом ∝ σ²) | розвідка, без PREREG і без вердикту: [README](fringes/README.md) |

Перевірка тверджень стартера `phs-soliton`:

```bash
uv run --no-project --with numpy python -I phs-soliton/starter/check_starter.py
```
