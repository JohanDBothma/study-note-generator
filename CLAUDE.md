# Study Notes PDF Generator

Generates clean, readable A4 PDF study sheets from Python data files.
Each sheet is a standalone script — define your data, pick a card style, call `render()`.

## Project structure

```
shared/
  styles.py   — colour palette, BADGE_COLORS, CSS, escape()
  cards.py    — card builder functions (one per PDF style)
  renderer.py — chart helpers, build_html(), render()
output/       — generated PDFs land here (gitignored)
```

## Adding a new study sheet

Create a new Python file at the project root:

```python
from shared.cards import build_preprocessing_card   # or build_algorithm_card
from shared.renderer import render

ITEMS = [{ ... }]   # see card schemas below

def main():
    render(ITEMS, "Title", "Subtitle",
           "output/my_sheet.pdf",
           card_builder=build_preprocessing_card)

if __name__ == "__main__":
    main()
```

Run with `python my_sheet.py` or `python my_sheet.py --output path/to/file.pdf --html`.

## Card schemas

### `build_algorithm_card`
For classifier / model comparison sheets with matplotlib charts.

```python
{
    "title":          str,
    "type":           str,           # badge label — must be a key in BADGE_COLORS
    "cliff":          str,           # one-paragraph plain-English summary
    "use_case_title": str,
    "use_case":       str,
    "math":           str,           # key equations, preformatted
    "use_when":       [str, ...],    # 3–4 bullet points
    "code":           str,           # code snippet
    "viz_label":      str,           # chart caption
    "chart_fn":       callable,      # zero-arg function returning a matplotlib figure
}
```

### `build_preprocessing_card`
For data wrangling steps with before/after dataset tables.

```python
{
    "title":          str,
    "type":           str,
    "cliff":          str,
    "use_case_title": str,
    "use_case":       str,
    "use_when":       [str, ...],
    "code":           str,
    "dataset": {
        "before": { "label": str, "columns": [...], "rows": [[...], ...] },
        "after":  { "label": str, "columns": [...], "rows": [[...], ...] },
    },
}
```

## Shared styling

All colours, badge types, and CSS live in `shared/styles.py`.
- Add new badge types to `BADGE_COLORS`
- Adjust fonts, spacing, or colours in `CSS`
- `escape()` is available from `shared.styles` for safe HTML interpolation
