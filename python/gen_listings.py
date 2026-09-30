"""
Project 2: split each dataset's raw scalation_output.txt into one listing file
per banner-delimited section, for \\lstinputlisting in the LaTeX report.

Banners in Project2.scala look like:
    ------------------------------------------------
    | [AutoMPG] Section 1: Regression (ScalaTion)   |
    ------------------------------------------------

This finds each "[Dataset] Section N..." banner and writes everything up to
the next banner (or end of file) to report/listings/<dataset>_sectionN<letter>.txt.
"""

import os
import re

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE, "results")
LISTINGS_DIR = os.path.join(BASE, "report", "listings")
os.makedirs(LISTINGS_DIR, exist_ok=True)

DATASETS = ["auto_mpg", "concrete", "airfoil"]

BANNER_RE = re.compile(
    r"^\| \[(?P<ds>\w+)\] Section (?P<sec>\d)(?P<letter>[a-z]?): (?P<title>.*?) \|\s*$",
    re.MULTILINE,
)


def split_one(name):
    path = os.path.join(RESULTS_DIR, name, "scalation_output.txt")
    if not os.path.exists(path):
        print(f"  (skip {name}: no scalation_output.txt)")
        return
    with open(path, encoding="utf-8-sig", errors="replace") as f:
        text = f.read()

    matches = list(BANNER_RE.finditer(text))
    if not matches:
        print(f"  (skip {name}: no section banners found)")
        return

    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip("\n")
        sec, letter = m.group("sec"), m.group("letter")
        out_name = f"{name}_section{sec}{letter}.txt"
        with open(os.path.join(LISTINGS_DIR, out_name), "w", encoding="utf-8") as f:
            f.write(f"{m.group('title')}\n" + "=" * len(m.group("title")) + "\n\n")
            f.write(body)
        print(f"  wrote {out_name}  ({m.group('title')})")


def main():
    for name in DATASETS:
        print(f"=== {name} ===")
        split_one(name)


if __name__ == "__main__":
    main()
