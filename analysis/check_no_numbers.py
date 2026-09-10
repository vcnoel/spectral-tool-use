"""Verify the abstract and introduction contain no generated numbers."""
import os
import re

os.chdir(r"c:\Users\valno\Dev\spectral-tool-use\paper\iclr")

MACRO = re.compile(r"\\([a-zA-Z]+)\{\}")
NUMBER = re.compile(r"(?<![A-Za-z])\d+\.\d+")
LATEX_OPT = re.compile(r"\[[^\]]*\]")   # layout options such as leftmargin=1.4em

# macros that carry a measured value, as opposed to structural ones
VALUE_PREFIXES = ("auc", "rate", "n", "pos", "mode", "perHead", "decorr",
                  "transfer", "fiedler", "norm", "comb")

targets = {
    "abstract (main.tex)": open("main.tex", encoding="utf-8").read()
        .split(r"\begin{abstract}")[1].split(r"\end{abstract}")[0],
    "introduction": open("sections/s1_intro.tex", encoding="utf-8").read(),
}

bad = 0
for name, text in targets.items():
    macros = [m for m in MACRO.findall(text)
              if any(m.startswith(p) for p in VALUE_PREFIXES)]
    decimals = NUMBER.findall(LATEX_OPT.sub("", text))
    status = "clean" if not macros and not decimals else "HAS NUMBERS"
    print(f"{name:22s} {status}")
    if macros:
        print("   value macros:", sorted(set(macros)))
    if decimals:
        print("   literal decimals:", decimals)
    bad += bool(macros or decimals)

print("\nOK" if not bad else "\nFAIL")
raise SystemExit(1 if bad else 0)
