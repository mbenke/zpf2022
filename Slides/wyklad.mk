# Makefile borrowed from github.com/bos/stanford-cs240h
DESTDIR=../../www
MDFILE := $(word 1, $(basename $(wildcard *.md)))
#MATHJAXURL='http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML'
MATHJAXURL='https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js'
L ?= $(MDFILE)
MATH=--mathjax=$(MATHJAXURL)
#MATH=--mathml
PDOPTS=$(MATH) #--self-contained
#PANDOC=~/.cabal/bin/pandoc $(PDOPTS)
PANDOC=pandoc $(PDOPTS)

all: $(DESTDIR)/$(L).html $(DESTDIR)/$(L)-slides.html
.PHONY: all echo

$(DESTDIR)/$(L).html: $(L).md
	@test -f $<
	$(PANDOC) -s -t html -o $@ $<

$(DESTDIR)/$(L)-slides.html: $(L).md Makefile # $(wildcard ./pandoc/slidy/*)
	@test -f $<
	$(PANDOC) -s -t slidy -o $@ $<

echo: 
	echo $(MDFILE)
