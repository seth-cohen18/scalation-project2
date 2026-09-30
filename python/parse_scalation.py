"""
Parse the verbose scalation_output.txt logs (one per dataset) down to the
concise numbers needed for the report: R^2/adjR^2/coefficients for each
section's model(s), forward-selection order, and symbolic-regression terms.

Writes project 2/results/<dataset>/scalation_digest.txt.
"""

import os
import re

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE, "results")
DATASETS = ["auto_mpg", "concrete", "airfoil"]

SECTION_RE = re.compile(r"^\| (\[\w+\] Section .*?) \|\s*$", re.MULTILINE)
FITMAP_RE = re.compile(r"fitMap\s+qof = LinkedHashMap\((.*?)\)")
FEATURES_RE = re.compile(r"features\s+fn\s+= Array\((.*?)\)")
PARAM_RE = re.compile(r"parameter\s+b\s+= VectorD\((.*?)\)")
ADDVAR_RE = re.compile(
    r"\| forwardSelAll: \(l = (\d+)\) (INITIAL|ADD) variable \((\d+), (\S+?)\)"
    r"(?: => cols = LinkedHashSet\(([^)]*)\)(?: @ ([\d.eE+-]+))?)?"
)
SELECTED_RE = re.compile(r"^(selected (?:columns|features) .*|columns added.*|features added.*|"
                          r"best step by.*|k = \d+ terms.*)$", re.MULTILINE)
QOF_KEYS = ["rSq", "rSqBar", "sse", "rmse", "smape", "aic"]


def fitmap_summary(block):
    m = FITMAP_RE.search(block)
    if not m:
        return None
    d = {}
    for pair in m.group(1).split(", "):
        k, v = pair.split(" -> ")
        d[k] = v
    return {k: d.get(k) for k in QOF_KEYS}


def digest_one(name):
    path = os.path.join(RESULTS_DIR, name, "scalation_output.txt")
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8-sig", errors="replace") as f:
        text = f.read()

    sections = list(SECTION_RE.finditer(text))
    out_lines = [f"=== {name}: ScalaTion digest ===\n"]

    for i, m in enumerate(sections):
        title = m.group(1)
        start = m.end()
        end = sections[i + 1].start() if i + 1 < len(sections) else len(text)
        block = text[start:end]

        out_lines.append(f"\n--- {title} ---")

        fm = fitmap_summary(block)
        if fm:
            out_lines.append("  QoF: " + ", ".join(f"{k}={v}" for k, v in fm.items() if v))

        fmatch = FEATURES_RE.search(block)
        pmatch = PARAM_RE.search(block)
        if fmatch and pmatch:
            feats = [s.strip() for s in fmatch.group(1).split(",")]
            params = [s.strip() for s in pmatch.group(1).split(",")]
            if len(feats) == len(params):
                out_lines.append("  Coefficients:")
                for fn, b in zip(feats, params):
                    out_lines.append(f"    {fn:>28s} = {b}")

        add_vars = ADDVAR_RE.findall(block)
        if add_vars:
            out_lines.append("  Forward selection order:")
            for l, kind, idx, nm, cols, metric in add_vars:
                metric_s = f" (metric={metric})" if metric else ""
                out_lines.append(f"    step {l}: {kind} {nm}{metric_s}")

        for sm in SELECTED_RE.finditer(block):
            out_lines.append("  " + sm.group(1).strip())

    out_path = os.path.join(RESULTS_DIR, name, "scalation_digest.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines))
    print(f"wrote {out_path}")


def main():
    for name in DATASETS:
        digest_one(name)


if __name__ == "__main__":
    main()
