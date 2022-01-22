root = main
dist = pdflatex
latexmkopts = -$(dist) -shell-escape

all: $(root).pdf
.PHONY: $(root).pdf 

$(root).pdf: $(root).tex
	latexmk $(latexmkopts) $(root)

clean:
	rm -rf `biber --cache`
	rm -rf _minted-$(root)
	latexmk -bibtex -CA

watch: clean
	latexmk $(latexmkopts) -pvc $(root)

quick: clean
	$(dist) -shell-escape $(root)

view:
	open $(root).pdf
