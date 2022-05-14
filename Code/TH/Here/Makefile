BINFILES=here

all: $(BINFILES)

here:
	ghc -o here --make Main.hs

clean:
	-rm -f *.hi *.o *.dyn_hi *.dyn_o $(BINFILES)
