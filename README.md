# lab

Експерименти. Кожен експеримент лежить в окремому каталозі: стартер (джерело та його транскрипція) плюс goal, за яким експеримент проводиться до вердикту.

| Каталог | Що це | Стан |
|---|---|---|
| [`phs-soliton/`](phs-soliton/) | Порт-гамільтонові та солітонні рекурентні резервуари проти GRU: [стартер](phs-soliton/starter/port-hamiltonian-soliton-reservoirs.md), [goal](phs-soliton/GOAL.md) | стартер перевірено (12/12), goal готовий, прогонів ще немає |

Перевірка тверджень стартера `phs-soliton`:

```bash
uv run --no-project --with numpy python -I phs-soliton/starter/check_starter.py
```
