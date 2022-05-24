# Example programs

Build/run the example programs in Code/Par e.g.

```
$ stack build
$ stack exec -- sudoku3b sudoku17.1000.txt +RTS -s -N2
```

or, if not using stack install the `parallel` package and

```
make
./sudoku3b sudoku17.1000.txt +RTS -s -N2
```

(you can also use the supplied cabal file and run `cabal build`)


Install threadscope e.g/ from https://github.com/haskell/ThreadScope/releases
(binaries available for Linux/Mac/Windows)

Run example programs with `-l` to generate eventlog, and analyze it with threadscope:

```
$ stack exec -- parfib +RTS -N2 -l -s
$ threadscope parfib.eventlog
```

# N queens

Write a function putting n queens on n*n chessboard

* sequential

* parallel

~~~~ {.haskell}
type PartialSolution = [Int]
type Solution = PartialSolution
type BoardSize = Int

queens :: BoardSize -> [Solution]
queens n = iterate (concatMap (addQueen n)) [[ ]] !! n

addQueen :: BoardSize -> PartialSolution -> [PartialSolution]
addQueen n s = [x : s | x <- [1..n], safe x s 1]

safe :: Int -> PartialSolution -> Int -> Bool
safe x [] n = True
safe x (c : y) n = x /= c && x /= c + n 
       && x /= c - n && safe x y (n + 1)
~~~~

Analyze with threadscope!

Try using an explicit solution tree and compare performance.

``` haskell
data Tree a = Leaf a | Node [Tree a]
instance NFData a => NFData (Tree a) where
  rnf (Leaf a) = rnf a
  rnf (Node ts) = rnf (map rnf ts)

-- | evaluating tree in parallel up to given depth
parTree :: NFData a => Int -> Strategy (Tree a)
parTree 0 tree = rdeepseq tree
parTree _ (Leaf a) = return (Leaf a)
parTree n (Node ts) = Node <$> parList (parTree (n-1)) ts
```

# Fibonacci

~~~~ {.haskell}
cutoff :: Int
cutoff = 20

parFib n | n < cutoff = fib n
parFib n = p `par` q `pseq` (p + q)
    where
      p = parFib $ n - 1
      q = parFib $ n - 2

fib n | n<2 = n
fib n = fib (n - 1) + fib (n - 2)
~~~~

* Rewrite parFib using the `Eval` monad

* Ditto using Strategies

* Check what cutoff values are best for different parallelism factors

* Try out other strategies


# Moar parallelism

* Try to parallelise your own example (primes? sorting? ...?)