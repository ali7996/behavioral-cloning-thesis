# Technical report

`main.tex` — *Behavioral Cloning for Self-Driving Cars: A Leak-Free Re-evaluation
and a Negative Result on Temporal Models*.

A short write-up of the methodology behind this repository: the corrected temporal
models, the leak-free contiguous-split evaluation, and the negative result on
recurrent architectures. Every number is taken from the saved metric files in
`models/`.

## Compile

No local LaTeX install needed — use [Overleaf](https://www.overleaf.com):

1. **New Project → Upload Project**, and upload `main.tex` and `refs.bib`.
2. Overleaf compiles automatically (pdfLaTeX + BibTeX). Set `main.tex` as the main file if prompted.

With a local TeX distribution instead:

```bash
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## Files

- `behavioral-cloning-report.pdf` — the compiled paper (ready to read)
- `main.tex` — the paper source
- `refs.bib` — bibliography (Bojarski 2016, Codevilla 2018/2019, Pomerleau 1989, Hochreiter & Schmidhuber 1997, Kingma & Ba 2015)
