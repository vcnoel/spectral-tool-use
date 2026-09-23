# paper/

Paper sources only. Everything that computes a number lives outside this
directory:

- `analysis/make_paper_numbers.py` builds `iclr/generated_numbers.tex` from
  the result files, so no number is typed into the text by hand.
- `analysis/exp_jensen_gap.py` measures the head-disagreement gap of
  Proposition 2.
- `analysis/exp_decorrelation.py` tests the prediction of Corollary 2.
- `run_pilot_v2.py` produces the runs; `make_paper_tables.py` produces
  `docs/RESULTS.md`.

Build:

```bash
python analysis/make_paper_numbers.py     # refresh the numbers
cd paper/iclr && latexmk -pdf main.tex
```

A macro with no backing result renders as a visible `[PENDING]` marker, and an
undefined macro stops the build.
