---
title: Advanced Functional Programming
subtitle: Parallelism and Concurrency
author:  Marcin Benke
date: May 24, 2022
---

# Parallelism vs concurrency

``A *parallel* program is one that uses a multiplicity of computational
hardware (e.g. multiple processor cores) in order to perform
computation more quickly.  Different parts of the computation are
delegated to different processors that execute at the same time (in
parallel), so that results may be delivered earlier than if the
computation had been performed sequentially.

In contrast, *concurrency* is a program-structuring technique in which
there are multiple threads of control. Notionally the threads of
control execute "at the same time"; that is, the user sees their
effects interleaved. Whether they actually execute at the same time or
not is an implementation detail; a concurrent program can execute on a
single processor through interleaved execution, or on multiple
physical processors.''

--- [Simon Marlow, *Parallel and Concurrent Programming in Haskell*](https://simonmar.github.io/pages/pcph.html)

(recommended reading, free to read online; NB ignore the "Start Your Free Trial" nags)

TL;DR:

* concurrent - result may depend on how the control threads interleave
* parallel - same result independently of interleavings

# Haskell parallelism

* Parallel programs in Haskell are *deterministic*
* The same pure program gives the same answer on 1 and N cores
* Programs can be debugged on 1 core
* No risk of deadlock or race conditions
* No need to worry about thread communication and synchronisation.

# Evaluation

Before discussing parallel evaluation, let's look at Haskell evaluation in general

```
> let x = 1+2::Int
> :sprint x
x = _
> x
3
> :sprint x
x = 3
```

NB do not omit `::Int` lest you get misled.

The `:sprint` command prints expression without evaluating; `_` means "unevaluated" (thunk).

![thunk](plusThunk.png "a thunk")

# Sharing

```
> let {x = 1+2::Int; y = (x,x) }
> :sprint y
y = (_,_)
> fst y
3
> :sprint vv
vv = (3,3)
```

Evaluating `fst vv` evaluated `x` which was both first and second component (was shared)

![Sharing](sharing.png "Sharing example")

# Forcing evaluation - seq

```
> let x = 1+2 :: Int
> let y = x+1
> :sprint x
x = _
> :sprint y
y = _
> seq y ()
()
> :sprint y
y = 4
> :sprint x
x = 3
```

We may think of `seq a b` as *sequentially* evaluating `a` then returning `b`.

A more precise description is that `seq a b` introduces a data dependency between `a` and its result (`b`): when the latter value is evaluated,  `a` must be evaluated as well.

OTOH it may happen that none of them is needed/evaluated, e.g.

```
> const 42 (undefined `seq` [1..])
42
```
# Normalisation and normal forms

* Normal Form - there are no redexes
* Head Normal Form - no redexes in the head, i.e. `\x1...xn -> xN1...Nk`
* Weak Head Normal Form - `xN1...Nk` or 

# WHNF - shallow evaluation

`seq` evaluates to so called Weak Head Normal Form
- more or less up to first constructor:

```
Prelude> let xs = map (+1) [1..10] :: [Int]
Prelude> :sprint xs
xs = _
Prelude> seq xs ()
()
Prelude> :sprint xs
xs = _ : _
```

BTW, similar thing happens when evaluating `case`:

```
Prelude> let xs = map (+1) [1..10] :: [Int]
Prelude> case xs of { [] -> (); _:_ -> () }
()
Prelude> :sprint xs
xs = _ : _
```

We will discuss deep evaluation (normal form) later.

# Parallel evaluation - `par`

`Control.Parallel.par` is supplied by the `parallel` package.

`par` and `seq` have much in common. The function par let you start a computation in parallel and seq forces a computation to actually take place

``` haskell
par :: a -> b -> b
seq :: a -> b -> b
```

* `seq a b` forces evaluation of `a` and returns b
* `par a b` initiates evaluation of `a` and returns `b` immediatly, not waiting until computation finishes

```
> import Control.Parallel
> let n = length [1..10^9]
> par n ()
()
> :sprint n
n = _
-- wait 10 seconds
> :sprint n
n = 1000000000
```

* by itself `par` evaluates shallowly:

```
> let xs = map (+1) [1..10] :: [Int]
> par xs ()
()
> :sprint xs
xs = _ : _
```

# `par`, `seq` and `pseq`

`seq` and `pseq` are almost equivalent, but differ in their runtime
behaviour in a subtle way: `seq` can evaluate its arguments in either
order, but `pseq` is required to evaluate its first argument before its
second, which makes it more suitable for controlling the evaluation
order in conjunction with `par`.

Consider

```
(a `par` b) `pseq` (a + b)
```

in this example we want to fully compute a and b (in parallel) before adding them.

# The Evaluation-order Monad

~~~~ {.haskell}
-- Control.Parallel.Strategies
data Eval a
instance Monad Eval

runEval :: Eval a -> a
rseq :: a -> Eval a  -- "eval argument, wait for result"
rpar :: a -> Eval a  -- "my argument can be evaluated in parallel"
~~~~

Note:

* the argument to rpar should be a thunk,
  otherwise nothing happens, because there is no work to perform in parallel.

`Eval` is basically just a strict identity monad:

```
data Eval a = Done a

runEval :: Eval a -> a
runEval (Done x) = x

instance Monad Eval where
  return x = Done x
  Done x >>= k = k x   -- Note: pattern 'Done x' makes '>>=' strict
```

Hence `rseq` and `rpar` guide the evaluation order,
but the computed value is the same it would be without them.

# deepseq & friends

deepseq: fully evaluates the first argument, before returning the second.

~~~~ {.haskell}
seq :: a -> b -> b
-- Control.DeepSeq
class NFData a where
    rnf :: a -> ()
-- rnf should reduce its argument to normal form
-- (that is, fully evaluate all sub-components),
-- and then return '()'

deepseq :: NFData a => a -> b -> b
deepseq a b = rnf a `seq` b

force ::  NFData a => a -> a
force x = deepseq x x

-- Control.Exception
-- Forces its argument to be evaluated to weak head normal form
-- when the resultant IO action is executed.
evaluate :: a -> IO a
~~~~

# deepseq & friends

`seq` forces ``shallow'' evaluation (just the root)

`deepseq` forces the  whole tree up to the leaves.

```
> let x = [undefined] :: [Int] in x `seq` length x
1

> let x = [undefined] :: [Int] in x `deepseq` length x
*** Exception: Prelude.undefined
```
Of course the forcing happens only when `deepseq` itself is evaluated

```
> let x = undefined::Int
> let y = deepseq x () in 42
42
> :sprint x
x = _
```

# Sudoku

An example with lots of computation: solving Sudoku

Every line of input contains an instance of the problem

Sequential program:

```haskell
main = do
    [f] <- getArgs
    grids <- fmap lines $ readFile f
    print $ length $ filter isJust $ map solve grids
```

```
$ ghc -O2 -threaded sudoku1.hs
$ ./sudoku1 sudoku17.1000.txt +RTS -s
  TASKS: 3 (1 bound, 2 peak workers (2 total), using -N1)
  SPARKS: 0 (0 converted, 0 overflowed, 0 dud, 0 GC'd, 0 fizzled)

  Total   time    2.53s  (  2.56s elapsed)
```

# Multicore?


~~~~
$ ghc -O2 -threaded sudoku1.hs
$ ./sudoku1 sudoku17.1000.txt +RTS -s
  TASKS: 3 (1 bound, 2 peak workers (2 total), using -N1)
  SPARKS: 0 (0 converted, 0 overflowed, 0 dud, 0 GC'd, 0 fizzled)

  Total   time    2.53s  (  2.56s elapsed)
~~~~

We can specify the number of threads using -N

~~~~
$ ./sudoku1 sudoku17.1000.txt +RTS -s -N16
  TASKS: 18 (1 bound, 17 peak workers (17 total), using -N16)
  SPARKS: 0 (0 converted, 0 overflowed, 0 dud, 0 GC'd, 0 fizzled)

  Total   time   16.84s  (  4.09s elapsed)
~~~~

Our program works slower - we unnecessarily start N-1 additional threads that only get in the way.

# A parallel program

~~~~ {.haskell}
import Control.Parallel.Strategies
main :: IO ()
main = do
    [f] <- getArgs
    grids <- fmap lines $ readFile f
    let (as,bs) = splitAt (length grids `div` 2) grids
    print (length (runEval (work as bs)))

work as bs =  do
       a <- rpar (force (map solve as))
       b <- rpar (force (map solve bs))
       return (filter isJust (a++b))
~~~~

We create two light threads, in GHC called ``sparks'' (these are not OS threads).


# Results

~~~~
$ ghc -O2 -rtsopts -threaded --make sudoku2.hs
$ ./sudoku2 sudoku17.1000.txt +RTS -N2 -s -RTS

  TASKS: 4 (1 bound, 3 peak workers (3 total), using -N2)
  SPARKS: 2 (1 converted, 0 overflowed, 0 dud, 0 GC'd, 1 fizzled)

  Total   time    2.73s  (  1.77s elapsed)
~~~~

Better, but we are still unable to use the whole power:

```
./sudoku2 sudoku17.1000.txt +RTS -N16 -s -RTS

  TASKS: 18 (1 bound, 17 peak workers (17 total), using -N16)
  SPARKS: 2 (1 converted, 0 overflowed, 0 dud, 0 GC'd, 1 fizzled)

  Total   time   15.12s  (  3.19s elapsed)
```

# Sparks

* A new spark is created with each use of `rpar` (and put in the spark pool)

* Whenever the system has a free computational unit (processor, core), it allocates it a new spark from the pool ("convert")

* Computational unit - Haskelll Execution Context (HEC)

* One unit always occupied by the main thread.

Spark creation may fail due to

* pool overflow

* expression in WHNF already (dud)

# Spark pool

sparks in the pool may be

* "converted" (allocated a HEC)

* evaluated out of order (fizzle)

* garbage collected (GC)

#

![spark lifecycle](spark-lifecycle800.png "Life cycle of a spark")

# sudoku1 vs sudoku2
~~~~
$ ./sudoku1 sudoku17.1000.txt +RTS -s
  TASKS: 3 (1 bound, 2 peak workers (2 total), using -N1)
  SPARKS: 0 (0 converted, 0 overflowed, 0 dud, 0 GC'd, 0 fizzled)

  Total   time    2.53s  (  2.56s elapsed)
~~~~

~~~~
$ ./sudoku2 sudoku17.1000.txt +RTS -N2 -s -RTS

  TASKS: 4 (1 bound, 3 peak workers (3 total), using -N2)
  SPARKS: 2 (1 converted, 0 overflowed, 0 dud, 0 GC'd, 1 fizzled)

  Total   time    2.73s  (  1.77s elapsed)
~~~~

Better, but still far from ideal.

# Threadscope

* A tool to analyse threaded program execution
* can be built from source, but easiest to use binaries - see https://hackage.haskell.org/package/threadscope
* `ghc -eventlog`
* executing program `prog` yields `prog.eventlog`

~~~~
$ stack ghc -- -O2 -threaded -eventlog sudoku2.hs
$ ./sudoku2 sudoku17.1000.txt +RTS -N2 -ls
$ threadscope sudoku2.eventlog &
~~~~

# Threadscope - sudoku2

![](sudoku2.png "sudoku2.eventlog")

Two work units of unequal size - some time wasted.

We need better work allocation (granularity).

# Dynamic work allocation

Parallel map: process list elements in parallel

~~~~ {.haskell}
parMap :: (a -> b) -> [a] -> Eval [b]
parMap f [] = return []
parMap f (a:as) = do
   b <- rpar (f a)
   bs <- parMap f as
   return (b:bs)
~~~~

Evaluation:

~~~~ {.haskell}
    runEval (parMap solve grids)
~~~~

NB no need to use `force` here - evaluating first constructor (`Nothing`/`Just`) is enough

# parMap - results (on students)

More efficient, easier to scale (we can use -N4, -N8 now)

```
$ ./sudoku3b sudoku17.1000.txt +RTS -N1 -s 2>&1 | grep Total
  Total   time    3.497s  (  3.551s elapsed)

$ ./sudoku3b sudoku17.1000.txt +RTS -N2 -s 2>&1 | grep Total
  Total   time    5.143s  (  2.642s elapsed)

$ ./sudoku3b sudoku17.1000.txt +RTS -N4 -s 2>&1 | grep Total
  Total   time    5.167s  (  1.364s elapsed)

$ ./sudoku3b sudoku17.1000.txt +RTS -N8 -s 2>&1 | grep Total
  Total   time    5.317s  (  0.755s elapsed)

$ ./sudoku3b sudoku17.1000.txt +RTS -N16 -s 2>&1 | grep Total
  Total   time    5.943s  (  0.487s elapsed)
```

BTW `-N` without argument uses all capabilities, try it, but not on students:

```
$ ./sudoku3b sudoku17.1000.txt +RTS -N
sudoku3b: failed to create OS thread: Cannot allocate memory
```

# Threadscope - sudoku3 -N2

![](sudoku3.png "sudoku3.eventlog")

# Threadscope - sudoku3 -N8

![](sudoku3-N8.png "sudoku3-N8.eventlog")

# Strategies

Additional abstraction layer built upon the `Eval` monad

~~~~ {.haskell}
type Strategy a = a -> Eval a
r0 :: Strategy a
r0 x = return x

rseq :: Strategy a
rseq x = x `pseq` return x

rpar :: Strategy a
rpar x = x `par` return x

rdeepseq :: NFData a => Strategy a
rdeepseq = rnf x `pseq` return x

using :: a -> Strategy a -> a
x `using` s = runEval (s x)
~~~~

The advantage is that  `using s` can be removed (almost) without changing semantics
(at worst, the program will be ``more defined'')

# Parallel list processing

~~~~ {.haskell}
parMap f xs = map f xs `using` parList rseq

-- Control.Parallel.Strategies
parList :: Strategy a -> Strategy [a]
parList strat [] = return []
parList strat (x:xs) = do
	x' <- rpar (x `using` strat)
	xs' <- parList strat xs
	return (x':xs)
~~~~

The reason `using` works at all is that Haskell is lazy

`map f xs` creates a thunk

```
x `using` s = runEval (s x)

parMap f xs
= map f xs `using` parList rseq
= runEval (parList rseq (map f xs))
~ case (map f xs) of
    [] -> [];
    (y:ys) -> runEval $ do ...
```

# Cautionary note

Although the sparks are really cheap, one should not create too many of them

~~~~ {.haskell}
parFib n | n < 2 = n
parFib n = p `par` q `pseq` (p + q)
    where
      p = parFib $ n - 1
      q = parFib $ n - 2

main :: IO ()
main = print $ parFib 40
~~~~

~~~~
@azor:
./badfib +RTS -N2 -s -RTS
  SPARKS: 165597322 (16 converted, 14860689 overflowed, 0 dud,
                     150628741 GC'd, 107876 fizzled)
  Total   time    7.18s  (  3.65s elapsed)
  Productivity  71.8% of total user, 141.5% of total elapsed

N60:
 SPARKS: 190193153 (61919 converted, 2556576 overflowed, 0 dud,
                    140401503 GC'd, 47173155 fizzled)
  Total   time   65.95s  (  1.28s elapsed)
  Productivity  47.8% of total user, 2461.5% of total elapsed
~~~~

# Spark lifecycle

![spark lifecycle](spark-lifecycle800.png "Life cycle of a spark")


# Better

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

~~~~
./parfib +RTS -N60 -s -RTS
 SPARKS: 118393 (42619 converted, 0 overflowed, 0 dud,
                 11241 GC'd, 64533 fizzled)

  Total   time   17.91s  (  0.33s elapsed)
  Productivity  98.5% of total user, 5291.5% of total elapsed

-N60, cutoff=15
  SPARKS: 974244 (164888 converted, 0 overflowed, 0 dud,
                  156448 GC'd, 652908 fizzled)
  Total   time   13.59s  (  0.28s elapsed)
  Productivity  97.6% of total user, 4746.9% of total elapsed
~~~~

# Exercise

* Rewrite parFib using the `Eval` monad

* Ditto using Strategies

* Check what cutoff values are best for different parallelism factors

* Try out other strategies

# Threadscope

~~~~
$ ghc -O2 -threaded -eventlog --make badfib.hs
$ ./badfib +RTS -N2 -ls
$ ~/.cabal/bin/threadscope badfib.eventlog
~~~~

![threadscope:badfib](badfib.png "Threadscope")

# Threadscope

~~~~
$ ghc -O2 -threaded -eventlog --make parfib.hs
$ ./parfib +RTS -N2 -ls
$ ~/.cabal/bin/threadscope parfib.eventlog
~~~~

![threadscope:badfib](parfib.png "Threadscope")


# Exercise

Write a function putting n queens on n*n chessboard

* sequential (see e.g. `Code/queens.hs`)

* parallel

* examine the execution on thradscope and consider resizing the work units

# The end

~~~~ {.haskell}

~~~~
