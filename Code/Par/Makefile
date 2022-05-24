BINFILES=parfib badfib sudoku1 sudoku2 sudoku3b
GHCOPTS=-O2 -rtsopts -threaded -eventlog
GHC=ghc

%: %.hs
	$(GHC) $(GHCOPTS) -o $@ --make $^

all: $(BINFILES)

clean:
	-rm -f *.hi *.o *~

veryclean: clean
	-rm -f $(BINFILES)

.PHONY: clean veryclean
