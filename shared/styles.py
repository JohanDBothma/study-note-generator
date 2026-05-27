# ── colour palette ─────────────────────────────────────────────────────────────
C_BLUE  = "#378add"
C_AMBER = "#ef9f27"
C_GREEN = "#34d399"
C_RED   = "#b41e1e"
C_GRAY  = "#888888"
C_QUERY = "#c03030"

BADGE_COLORS = {
    # algorithm cards
    "Probability-based classifier": ("#fce7f3", "#9d174d"),
    "Linear classifier":            ("#dbeafe", "#1e40af"),
    "Rule-based tree":              ("#fef3c7", "#92400e"),
    "Instance-based learner":       ("#ede9fe", "#5b21b6"),
    "Maximum margin classifier":    ("#e0f2fe", "#0c4a6e"),
    "Deep learning":                ("#fff1f2", "#9f1239"),
    "Ensemble — parallel trees":    ("#dcfce7", "#166534"),
    "Ensemble — sequential trees":  ("#dcfce7", "#166534"),
    "Discriminant analysis":        ("#f0fdf4", "#14532d"),
    # preprocessing cards
    "Categorical encoding":         ("#fef3c7", "#92400e"),
    "Numerical scaling":            ("#dbeafe", "#1e40af"),
    "Feature engineering":          ("#ede9fe", "#5b21b6"),
    "Missing data":                 ("#fff1f2", "#9f1239"),
    "Dimensionality reduction":     ("#dcfce7", "#166534"),
}


def escape(text):
    return (str(text)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;"))


CSS = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

@page { size: A4; margin: 15mm 12mm 15mm 12mm; }

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
    --bg:      #ffffff;
    --surface: #ffffff;
    --border:  #e0ddd6;
    --text:    #1a1917;
    --muted:   #6b6860;
    --mono:    'IBM Plex Mono', monospace;
    --sans:    'IBM Plex Sans', sans-serif;
    --code-bg: #1c1b18;
    --code-fg: #e8e4d9;
}

body {
    font-family: var(--sans);
    background: var(--bg);
    color: var(--text);
    font-size: 10pt;
    line-height: 1.55;
}

header {
    border-bottom: 2pt solid var(--text);
    padding-bottom: 8pt;
    margin-bottom: 14pt;
}
header h1 { font-size: 18pt; font-weight: 600; letter-spacing: -0.02em; }
header p  { color: var(--muted); font-size: 9pt; margin-top: 2pt; font-weight: 300; }

.algo {
    background: var(--surface);
    border: 0.5pt solid var(--border);
    border-radius: 5pt;
    margin-bottom: 12pt;
    page-break-inside: avoid;
    overflow: hidden;
    width: 100%;
}

/* ── header row ── */
.algo-header {
    padding: 7pt 10pt 6pt;
    border-bottom: 0.5pt solid var(--border);
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8pt;
}
.algo-title {
    font-size: 12pt;
    font-weight: 600;
    letter-spacing: -0.01em;
    white-space: nowrap;
}
.badge {
    font-size: 7.5pt;
    font-weight: 500;
    padding: 2pt 8pt;
    border-radius: 20pt;
    white-space: nowrap;
    font-family: var(--mono);
    flex-shrink: 0;
}

/* ── cliff note: full width ── */
.cliff-row {
    border-bottom: 0.5pt solid var(--border);
    padding: 8pt 10pt;
}

/* ── use case + use-when: side by side ── */
.info-row {
    display: table;
    width: 100%;
    table-layout: fixed;
    border-bottom: 0.5pt solid var(--border);
}
.cliff-col, .use-col {
    display: table-cell;
    width: 50%;
    vertical-align: top;
    padding: 8pt 10pt;
}
.cliff-col { border-right: 0.5pt solid var(--border); }

/* ── two-column meta row (used by preprocessing cards) ── */
.meta-row {
    display: table;
    width: 100%;
    table-layout: fixed;
    border-bottom: 0.5pt solid var(--border);
}
.math-col, .when-col {
    display: table-cell;
    width: 50%;
    vertical-align: top;
    padding: 8pt 10pt;
}
.math-col { border-right: 0.5pt solid var(--border); }

/* ── full-width rows ── */
.math-row, .code-row {
    border-bottom: 0.5pt solid var(--border);
    padding: 8pt 10pt;
}


/* ── row 4: visual full width ── */
.viz-row { padding: 8pt 10pt; border-bottom: 0.5pt solid var(--border); }

/* ── shared ── */
.section-label {
    font-size: 7pt;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 4pt;
}

.cliff-note {
    font-size: 9pt;
    line-height: 1.6;
    color: var(--text);
    font-weight: 300;
}

.use-case {
    font-size: 8.5pt;
    line-height: 1.55;
    color: var(--muted);
    font-weight: 300;
    border-left: 2pt solid var(--border);
    padding-left: 7pt;
}
.use-case strong {
    color: var(--text);
    font-weight: 500;
    display: block;
    margin-bottom: 1pt;
}

.math-note {
    font-size: 8.5pt;
    line-height: 1.55;
    color: var(--muted);
    font-weight: 300;
    margin-top: 5pt;
}

.math-block {
    font-family: var(--mono);
    font-size: 7.5pt;
    line-height: 1.75;
    color: var(--text);
    white-space: pre;
}

.when-list {
    list-style: none;
    padding: 0;
    margin: 0;
    display: flex;
    flex-wrap: wrap;
    gap: 2pt 16pt;
}
.when-list li {
    font-size: 8pt;
    line-height: 1.5;
    color: var(--muted);
    font-weight: 300;
    padding-left: 11pt;
    position: relative;
}
.when-list li::before {
    content: "✓";
    position: absolute;
    left: 0;
    color: #4ade80;
    font-size: 8pt;
}

pre {
    background: var(--code-bg);
    color: var(--code-fg);
    font-family: var(--mono);
    font-size: 7.5pt;
    line-height: 1.65;
    padding: 8pt 10pt;
    border-radius: 4pt;
    white-space: pre;
    overflow: hidden;
}

.chart-wrap {
    border-radius: 4pt;
    text-align: center;
}
.chart-wrap img {
    width: 100%;
    height: auto;
    display: block;
}

/* ── dataset tables (preprocessing cards) ── */
.data-table {
    width: 100%;
    border-collapse: collapse;
    font-family: var(--mono);
    font-size: 7.5pt;
}
.data-table th {
    background: #f5f4f0;
    color: var(--muted);
    font-weight: 500;
    text-align: left;
    padding: 3pt 6pt;
    border-bottom: 0.5pt solid var(--border);
    white-space: nowrap;
}
.data-table td {
    padding: 2.5pt 6pt;
    border-bottom: 0.3pt solid var(--border);
    color: var(--text);
}
.data-table tbody tr:last-child td { border-bottom: none; }
"""
