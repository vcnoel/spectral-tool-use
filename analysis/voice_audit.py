"""
Voice audit of a LaTeX manuscript (research-writing, section 2), run from this file.

Reads every .tex file under a paper directory (numbers.tex excluded), removes comments,
then for prose lines:
  semicolons, "--" dashes and unicode en/em dashes      (outside math and control sequences)
  \\emph, \\textit, \\textbf                               (decorative italic and bold)
  section 23 phrases ("not tested", "future work", ...)
  ", not " and ", but " cadences                         (reported for reading, not counted as errors)
  sentence length (mean, median, count over 40 words) and paragraphs carrying more than five
  numbers after macro expansion (numbers.tex is read to expand macros).
A self-test runs first: each rule must fire on a line built to violate it and stay silent on a
clean line, so a pattern that silently matches nothing is caught.

Usage: python analysis/voice_audit.py paper/icml_v2      (exit 1 when an error-class rule fires)
"""
import re
import statistics
import sys
from pathlib import Path

BS = chr(92)
PHRASES = ["not tested", "stops short", "we do not", "is left to", "future work", "beyond the scope",
           "remains open", "the obvious", "first experiment to add", "not yet run", "pending"]
MATH = re.compile(r"\$[^$]*\$")
CS_ARG = re.compile(re.escape(BS) + r"(?:cref|Cref|ref|label|cite[pt]?|citet|citep|input|includegraphics|"
                    r"begin|end|texttt|url|wscitet|wscitep|caption|paragraph|section|subsection)\*?(\[[^\]]*\])?\{[^}]*\}")
CS = re.compile(re.escape(BS) + r"[A-Za-z@]+\*?")
NUM = re.compile(r"(?<![\w.\-])[-+]?\d+(?:\.\d+)?(?![\w])")


def strip_comments(line):
    return re.sub(r"(?<!" + re.escape(BS) + r")%.*", "", line)


def prose(line):
    t = MATH.sub(" ", line)
    t = CS_ARG.sub(" ", t)
    return t


def rules(line):
    """Error-class hits on one source line."""
    hits = []
    p = prose(line)
    pc = CS.sub(" ", p)
    if ";" in pc:
        hits.append("semicolon")
    if "--" in pc or "–" in pc or "—" in pc:
        hits.append("dash")
    for cmd in ("emph", "textit", "textbf"):
        if BS + cmd + "{" in line:
            hits.append("decoration")
    low = pc.lower()
    for ph in PHRASES:
        if ph in low:
            hits.append("phrase:" + ph)
    return hits


def cadence(line):
    pc = CS.sub(" ", prose(line))
    out = []
    if re.search(r",\s+not\s", pc):
        out.append(", not")
    if re.search(r",\s+but\s", pc):
        out.append(", but")
    return out


def selftest():
    bad = {"semicolon": "One clause; another clause.", "dash": "A range 1--2 here.",
           "decoration": "A " + BS + "emph{stressed} word.", "phrase:future work": "This is future work."}
    for rule, line in bad.items():
        assert rule in rules(line), (rule, rules(line))
    clean = ["A tool-calling model with $a;b$ in math and " + BS + "cref{sec:x-y} refs.",
             "% a comment; with -- dashes"]
    for line in clean:
        assert not rules(strip_comments(line)), (line, rules(line))
    assert ", not" in cadence("It tracks errors, not hardness.")
    print("self-test passed")


def expand(text, macros):
    def rep(m):
        return macros.get(m.group(1), m.group(0))
    for _ in range(3):
        text = re.sub(re.escape(BS) + r"([A-Za-z]+)\{\}", rep, text)
        text = re.sub(re.escape(BS) + r"([A-Za-z]+)(?![A-Za-z{])", rep, text)
    return text


def main(root):
    selftest()
    root = Path(root)
    macros = {}
    nt = root / "numbers.tex"
    if nt.exists():
        pat = re.compile("^" + re.escape(BS + "newcommand{" + BS) + r"(\w+)\}\{(.*)\}$")
        for l in nt.read_text(encoding="utf-8").splitlines():
            m = pat.match(l.strip())
            if m:
                macros[m.group(1)] = m.group(2)
    files = sorted(p for p in root.rglob("*.tex") if p.name != "numbers.tex")
    errors, cad, sents, dense = 0, [], [], []
    for f in files:
        lines = f.read_text(encoding="utf-8").splitlines()
        for i, raw in enumerate(lines, 1):
            line = strip_comments(raw)
            if not line.strip():
                continue
            for h in rules(line):
                errors += 1
                print(f"  {h:28s} {f.relative_to(root)}:{i}  {line.strip()[:90]}")
            for c in cadence(line):
                cad.append((c, f"{f.relative_to(root)}:{i}"))
            if f.name.startswith("table_") or line.lstrip().startswith(BS) and not line.lstrip().startswith(BS + "paragraph"):
                continue
            body = CS.sub(" ", prose(expand(line, macros)))
            for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", body):
                w = len(s.split())
                if w >= 3:
                    sents.append((w, f"{f.relative_to(root)}:{i}"))
            nums = NUM.findall(prose(expand(line, macros)).replace(",", " "))
            if len(nums) > 5:
                dense.append((len(nums), f"{f.relative_to(root)}:{i}"))
    print(f"\nerror-class hits: {errors}")
    print(f"cadences to read: {len(cad)}  " + ", ".join(f"{c} {w}" for c, w in cad))
    if sents:
        ws = [w for w, _ in sents]
        long_ = [(w, where) for w, where in sents if w > 40]
        print(f"sentences: {len(ws)}, mean {statistics.mean(ws):.1f} words, median {statistics.median(ws):.0f}, "
              f"over 40 words: {len(long_)}  " + ", ".join(f"{w}@{where}" for w, where in long_[:12]))
    print(f"paragraphs with more than five numbers: {len(dense)}  " + ", ".join(f"{n}@{w}" for n, w in dense))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "paper/icml_v2"))
