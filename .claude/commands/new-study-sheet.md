Create a new study sheet generator file in the `generators/` folder of this project.

## Step 1 — gather requirements

Ask the user the following three questions before doing anything else (ask all at once):

1. What is the **title** of the study sheet? (e.g. "Python Decorators", "Random Forest", "SQL Joins")
2. Which **card type** should it use?
   - `algorithm` — for ML model reference sheets (cliff note, use case, math, use-when bullets, chart, code)
   - `code` — for code walkthrough sheets (description, optional dataset table, syntax-highlighted code)
   - `terminology` — for definition/glossary sheets (term, definition, example, contrast) — note: card builder not yet implemented, scaffold only
3. Which **subfolder** under `generators/` should it go in? (e.g. `classification`, `python`, `sql`) — create a new one if it doesn't exist yet.

## Step 2 — derive names

From the title and subfolder, derive:
- `snake_case` filename, e.g. "Python Decorators" → `python_decorators`
- A short subtitle (one sentence describing what the sheet covers)
- The PDF output path: `pdfs/<subfolder>/<snake_case>.pdf`
- The module run command: `python -m generators.<subfolder>.<snake_case>`

If the subfolder is new, create `generators/<subfolder>/__init__.py` (empty file) so it works as a Python package. The `generators/__init__.py` already exists.

## Step 3 — generate the file

Create `generators/<snake_case>.py` using the matching template below.

---

### Template: `algorithm` card type

```python
#!/usr/bin/env python3
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
"""
<Title> — Study Sheet
=====================
<Subtitle>

Usage (run from project root):
    python -m generators.<snake_case>
    python -m generators.<snake_case> --output pdfs/<snake_case>.pdf
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from shared.styles import C_BLUE, C_AMBER, C_GREEN, C_RED, C_GRAY, C_QUERY
from shared.renderer import fig_to_b64, rng_seq, new_fig, style_ax, render
from shared.cards import build_algorithm_card


# ── charts ────────────────────────────────────────────────────────────────────

def chart_example():
    fig, ax = new_fig()
    style_ax(ax)
    ax.text(0.5, 0.5, "Replace with real chart", ha="center", va="center",
            transform=ax.transAxes, fontsize=12, color=C_GRAY)
    fig.tight_layout(pad=0.4)
    return fig_to_b64(fig)


# ── data ──────────────────────────────────────────────────────────────────────

ITEMS = [
    {
        "title": "Example Algorithm",
        "type": "Linear classifier",          # must be a key in shared/styles.py BADGE_COLORS
        "cliff": "One-paragraph plain-English summary of what this algorithm does.",
        "use_case_title": "Example use case title",
        "use_case": "Describe a concrete real-world scenario where this algorithm fits.",
        "math": "Explain how it works mathematically in plain English — no LaTeX needed.",
        "use_when": [
            "First condition where you'd reach for this algorithm",
            "Second condition",
            "Third condition",
            "Fourth condition",
        ],
        "code": """\
# Replace with real example code
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
""",
        "viz_label": "Description of what the chart shows",
        "chart_fn": chart_example,
    },
]


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate <title> study sheet PDF")
    parser.add_argument("--output", default="pdfs/<snake_case>.pdf")
    parser.add_argument("--html", action="store_true")
    args = parser.parse_args()

    render(
        ITEMS,
        title="<Title>",
        subtitle="<Subtitle>",
        output_path=args.output,
        save_html=args.html,
        card_builder=build_algorithm_card,
        show_header=True,
    )


if __name__ == "__main__":
    main()
```

---

### Template: `code` card type

```python
#!/usr/bin/env python3
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
"""
<Title> — Study Sheet
=====================
<Subtitle>

Usage (run from project root):
    python -m generators.<snake_case>
    python -m generators.<snake_case> --output pdfs/<snake_case>.pdf
"""

import argparse
from shared.cards import build_code_card
from shared.renderer import render


ITEMS = [
    {
        "title": "Example topic",
        "type": "Python built-in",            # must be a key in shared/styles.py BADGE_COLORS
        "description": (
            "Plain-English explanation of what this code does and WHY it works this way. "
            "Focus on the reasoning, not just restating the code."
        ),
        # "dataset_md": SAMPLE_DATA,          # optional — uncomment and define SAMPLE_DATA above
        "code": """\
# Replace with real annotated code
x = 1
""",
    },
]


def main():
    parser = argparse.ArgumentParser(description="Generate <title> study sheet PDF")
    parser.add_argument("--output", default="pdfs/<snake_case>.pdf")
    parser.add_argument("--html", action="store_true")
    args = parser.parse_args()

    render(
        ITEMS,
        title="<Title>",
        subtitle="<Subtitle>",
        output_path=args.output,
        save_html=args.html,
        card_builder=build_code_card,
        show_header=True,
    )


if __name__ == "__main__":
    main()
```

---

### Template: `terminology` card type

```python
#!/usr/bin/env python3
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
"""
<Title> — Study Sheet
=====================
<Subtitle>

Usage (run from project root):
    python -m generators.<snake_case>
    python -m generators.<snake_case> --output pdfs/<snake_case>.pdf

NOTE: build_terminology_card is not yet implemented in shared/cards.py.
      Add the card builder before running this generator.
"""

import argparse
from shared.cards import build_terminology_card
from shared.renderer import render


ITEMS = [
    {
        "title": "Example term",
        "type": "Definition",               # update BADGE_COLORS in shared/styles.py if needed
        "definition": "What the term means in plain English.",
        "example": "A concrete example that makes the definition tangible.",
        "contrast": "How this differs from a commonly confused term.",
    },
]


def main():
    parser = argparse.ArgumentParser(description="Generate <title> study sheet PDF")
    parser.add_argument("--output", default="pdfs/<snake_case>.pdf")
    parser.add_argument("--html", action="store_true")
    args = parser.parse_args()

    render(
        ITEMS,
        title="<Title>",
        subtitle="<Subtitle>",
        output_path=args.output,
        save_html=args.html,
        card_builder=build_terminology_card,
        show_header=True,
    )


if __name__ == "__main__":
    main()
```

---

## Step 4 — confirm

After creating the file, tell the user:
- The file path created
- The command to run it (`python -m generators.<snake_case>`)
- Any available `BADGE_COLORS` keys they can use for the `type` field (read them from `shared/styles.py`)
- If they chose `terminology`, remind them that `build_terminology_card` still needs to be implemented in `shared/cards.py`
