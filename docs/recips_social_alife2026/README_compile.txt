ALIFE 2026 paper package

Files:
- main.tex: paper source using the ALIFE 2026 template style file
- main.pdf: compiled paper
- refs.bib: bibliography database
- main.bbl: compiled bibliography (included so pdflatex can compile even if BibTeX is unavailable)
- alifeconf.sty: template style file
- figures/: generated figures used in the paper
- social-foodshare-compare.gif, social-corridor-compare.gif: supplementary visual demonstrations

Compile notes:
1. If main.bbl is present, pdflatex main.tex (run twice) should be enough.
2. To fully regenerate the bibliography, run:
   pdflatex main.tex
   bibtex main
   pdflatex main.tex
   pdflatex main.tex

This package was drafted against the code/results in recips-social and aligned to the shipped deterministic paper-profile results.
