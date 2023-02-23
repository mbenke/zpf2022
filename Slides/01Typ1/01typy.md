---
title: Advanced Functional Programming
author:  Marcin Benke
date: Feb 28, 2023
---

<meta name="duration" content="80" />

# Course plan
* Types and type classes
    * Algebraic types and type classes
    * Constructor classes
    * Multiparameter classes, functional dependencies
* Testing (QuickCheck)
* Dependent types, Agda, Idris, Coq, proving properties (ca 7 weeks)
* Dependent types in Haskell
    * Type families, associated types, GADTs
    * data kinds, kind polymorphism
* Metaprogramming
* Parallel and concurrent programming in Haskell
    * Multicore and multiprocessor programming (SMP)
    * Concurrency
    * Data Parallel Haskell
* Project presentations

Any wishes?

# Passing the course (Zasady zaliczania)
* Lab: fixed Coq project, student-defined simple Haskell project (group projects are encouraged)
* Oral exam, most important part of which is project presentation
* Alternative to Haskell project: presentation on interesting Haskell topics during the lecture (possibly plus lab)
    * Anyone interested?

# Installing GHC on your machine

Simplest way - `ghcup`: `https://www.haskell.org/ghcup/` np.

```
curl --proto '=https' --tlsv1.2 -sSf https://get-ghcup.haskell.org | sh
```

Alternatively you can use `stack`: http://haskellstack.org np.

```
curl -sSL https://get.haskellstack.org/ | sh
stack setup
stack ghci
```

(but cabal has improved a lot, so there is less reasons to use stack these days)



# Functional languages
* dynamically typed, strict, impure: e.g. Lisp
* statically typed, strict, impure: e.g. ML
* staticaly typed, lazy, pure: e.g. Haskell

This course: Haskell, focusing on types.

Rich type structure distinguishes Haskell among other languages.

Second part of the course: Coq, Idris where the type structure is arguably even richer.

# Types as a specification language

A function type often specifies not only its input and output but also relationship between them:


~~~~ {.haskell}
f :: forall a. a -> a
f x = ?
~~~~

If `f x` gives a result, it must be `x`

* Philip Wadler ["Theorems for Free"](http://ecee.colorado.edu/ecen5533/fall11/reading/free.pdf)

* `h :: a -> IO b` constructs a computation with possible side effects

    ~~~~ {.haskell}
    import Data.IORef

    f :: Int -> IO (IORef Int)
    f i = do
      print i
      r <- newIORef i
      return r

    main = do
      r <- f 42
      j <- readIORef r
      print j
    ~~~~



# Types as a specification language (2)

`g :: Integer -> Integer` may not have side effects visible outside

It may have local side effects

Example: Fibonacci numbers in constant memory

~~~~ {.haskell}
import Control.Monad.ST
import Data.STRef
fibST :: Integer -> Integer
fibST n =
    if n < 2 then n else runST fib2 where
      fib2 =  do
        x <- newSTRef 0
        y <- newSTRef 1
        fib3 n x y

      fib3 0 x _ = readSTRef x
      fib3 n x y = do
              x' <- readSTRef x
              y' <- readSTRef y
              writeSTRef x y'
              writeSTRef y (x'+y')
              fib3 (n-1) x y
~~~~

How come?

~~~~
runST :: (forall s. ST s a) -> a
~~~~

The type of `runST` guarantees that side effects do not leak;
`fibST` is pure.


# Types as a design language

* Designing programs using types and `undefined`

Define CNF formulas as:

``` haskell
type Literal = Int       -- variable id
type Clause = [Literal]  -- disjunction of literals
type Form = [Clause]     -- conjunction of clauses
type Model = [Literal]   -- true literals
```

We want to write a simple DPLL (Davis–Putnam–Logemann–Loveland) SAT solver:


1. If there's a unit clause `[l]`, `l` must be true - resolve `l` and continue
2. Otherwise choose any literal `l` and split on it: try resolutions with `l` and its negation

where resolving for `l` means simplifying the formula based on its known truth value:

```
True  || a = True
False || a = a
True  && b = b
```

# Types as a design language (2)

We can write down this design using types and undefined and later fill in the details

``` haskell
dpll :: Form -> Model -> Maybe Model
dpll [] model = Just model   -- empty conjunction trivially satisfiable
dpll cs model
  | [] `elem` cs = Nothing   -- empty disjunction trivially unsatisfiable
  | otherwise = maybe split go (findUnit cs)
  -- case findUnit cs of Nothing -> split; Just m -> go m
  where
    go :: Literal -> Maybe Model
    go l = dpll (resolve l cs) (l : model)
    split = go dlit <|> go (- dlit)
    dlit = anyLiteral cs

anyLiteral :: Form -> Literal
anyLiteral = undefined

findUnit :: Form -> Maybe Literal
findUnit = undefined

resolve :: Literal -> Form -> Form
resolve = undefined
```

# Typed holes

Newer Haskell version allow for typed holes that can be used instead of undefined,  e.g. with

```
findUnitClauses :: Clauses -> Clauses
headMaybe :: [a] -> Maybe a

findUnitClause :: Clauses -> Maybe Clause
findUnitClause  = headMaybe . _
```

we get

```
   • Found hole: _ :: Clauses -> [Clause]
    • In the second argument of ‘(.)’, namely ‘_’
      In the expression: headMaybe . _
      In an equation for ‘findUnitClause’: findUnitClause = headMaybe . _
    • Relevant bindings include
        findUnitClause :: Clauses -> Maybe Clause
          (bound at /home/ben/var/haskell/sat1/Berger3.hs:38:1)
      Valid hole fits include
        findUnitClauses :: Clauses -> Clauses
          (bound at /home/ben/var/haskell/sat1/Berger3.hs:41:1)
        cycle :: forall a. [a] -> [a]
        ...
```

# Types as a programming language

* Functions on types computed at compile time

    ~~~~ {.haskell}
    data Zero
    data Succ n

    type One   = Succ Zero
    type Two   = Succ One
    type Three = Succ Two
    type Four  = Succ Three

    one   = undefined :: One
    two   = undefined :: Two
    three = undefined :: Three
    four  = undefined :: Four

    class Add a b c | a b -> c where
      add :: a -> b -> c
      add = undefined
    instance              Add  Zero    b  b
    instance Add a b c => Add (Succ a) b (Succ c)
    ~~~~

    ~~~~
    *Main> :t add three one
    add three one :: Succ (Succ (Succ (Succ Zero)))
    ~~~~

**Exercise:** extend with multiplication and factorial

# Types as a programming language (2)

Vectors using type classes:

~~~~ {.haskell}
data Vec :: * -> * -> * where
  VNil :: Vec Zero a
  (:>) :: a -> Vec n a -> Vec (Succ n) a

vhead :: Vec (Succ n) a -> a
vhead (x :> xs) = x
~~~~

**Exercise:** write `vtail`, `vlast`

We would like to have

~~~~ {.haskell}
vappend :: Add m n s => Vec m a -> Vec n a -> Vec s a
~~~~

but here the base type system is too weak

# Types as a programming language (3)

* Vectors with type families:

    ~~~~ {.haskell}
    data Zero = Zero
    data Suc n = Suc n

    type family m :+ n
    type instance Zero :+ n = n
    type instance (Suc m) :+ n = Suc(m:+n)

    data Vec :: * -> * -> * where
      VNil :: Vec Zero a
      (:>) :: a -> Vec n a -> Vec (Suc n) a

    vhead :: Vec (Suc n) a -> a
    vappend :: Vec m a -> Vec n a -> Vec (m:+n) a
    ~~~~


# Dependent types

Real type-level programming and proving properties is possible in a language with dependent types, such as Agda or Idris:

~~~~
module Data.Vec where
infixr 5 _∷_

data Vec (A : Set a) : ℕ → Set where
  []  : Vec A zero
  _∷_ : ∀ {n} (x : A) (xs : Vec A n) → Vec A (suc n)

_++_ : ∀ {a m n} {A : Set a} → Vec A m → Vec A n → Vec A (m + n)
[]       ++ ys = ys
(x ∷ xs) ++ ys = x ∷ (xs ++ ys)

module UsingVectorEquality {s₁ s₂} (S : Setoid s₁ s₂) where
  xs++[]=xs : ∀ {n} (xs : Vec A n) → xs ++ [] ≈ xs
  xs++[]=xs []       = []-cong
  xs++[]=xs (x ∷ xs) = SS.refl ∷-cong xs++[]=xs xs
~~~~


# A problem with dependent types

While Haskell is sometimes hard to read, dependent types are even easier to overdo:

~~~~
  now-or-never : Reflexive _∼_ →
                 ∀ {k} (x : A ⊥) →
                 ¬ ¬ ((∃ λ y → x ⇓[ other k ] y) ⊎ x ⇑[ other k ])
  now-or-never refl x = helper <$> excluded-middle
    where
    open RawMonad ¬¬-Monad

    not-now-is-never : (x : A ⊥) → (∄ λ y → x ≳ now y) → x ≳ never
    not-now-is-never (now x)   hyp with hyp (, now refl)
    ... | ()
    not-now-is-never (later x) hyp =
      later (♯ not-now-is-never (♭ x) (hyp ∘ Prod.map id laterˡ))

    helper : Dec (∃ λ y → x ≳ now y) → _
    helper (yes ≳now) = inj₁ $ Prod.map id ≳⇒ ≳now
    helper (no  ≵now) = inj₂ $ ≳⇒ $ not-now-is-never x ≵now
~~~~

...even though writing such proofs is fun.

# Parallel Haskell

Parallel Sudoku solver

~~~~ {.haskell}
main = do
    [f] <- getArgs
    grids <- fmap lines $ readFile f
    runEval (parMap solve grids) `deepseq` return ()

parMap :: (a -> b) -> [a] -> Eval [b]
parMap f [] = return []
parMap f (a:as) = do
   b <- rpar (f a)
   bs <- parMap f as
   return (b:bs)

solve :: String -> Maybe Grid
~~~~

~~~~
$ ./sudoku3b sudoku17.1000.txt +RTS -N2 -s -RTS
  TASKS: 4 (1 bound, 3 peak workers (3 total), using -N2)
  SPARKS: 1000 (1000 converted, 0 overflowed, 0 dud, 0 GC'd, 0 fizzled)

  Total   time    2.84s  (  1.49s elapsed)
  Productivity  88.9% of total user, 169.6% of total elapsed

-N8: Productivity  78.5% of total user, 569.3% of total elapsed
N16: Productivity  62.8% of total user, 833.8% of total elapsed
N32: Productivity  43.5% of total user, 1112.6% of total elapsed
~~~~

# Parallel Fibonacci

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

# Data Parallel Haskell


~~~~ {.haskell}
{-# LANGUAGE ParallelArrays #-}
{-# OPTIONS_GHC -fvectorise #-}

module DotP where
import qualified Prelude
import Data.Array.Parallel
import Data.Array.Parallel.Prelude
import Data.Array.Parallel.Prelude.Double as D

dotp_double :: [:Double:] -> [:Double:] -> Double
dotp_double xs ys = D.sumP [:x * y | x <- xs | y <- ys:]
~~~~

Looks like list operations, but works on vectors and "automagically"
parallellises to any number of cores (also CUDA)


# Types in Haskell

* base types: `zeroInt :: Int`
* function types: `plusInt :: Int -> Int -> Int`
* polymorphic types `id :: a -> a`

    ~~~~ {.haskell}
    {-# LANGUAGE ExplicitForAll #-}
    g :: forall b.b -> b
    ~~~~

* constrained types `0 :: Num a => a`
* algebraic types

    ~~~~ {.haskell}
    data Tree a = Leaf | Node a (Tree a) (Tree a)
    ~~~~

* `Leaf`, `Node` are *value constructors

    ~~~~ {.haskell}
    data Tree a where
    	 Leaf :: Tree a
         Node :: a -> Tree a -> Tree a -> Tree a
    ~~~~

* `Tree` is a *type constructor*, an operation on types

* NB empty types are allowed:

    ~~~~ {.haskell}
    data Zero
    ~~~~

# Polymorphic typing

* Generalisation:

$${\Gamma \vdash e :: t, a \notin FV( \Gamma )}\over {\Gamma \vdash e :: \forall a.t}$$

 <!--
Jeśli $\Gamma \vdash e :: t, a \notin FV( \Gamma )$

to $\Gamma \vdash e :: \forall a.t$

  Γ ⊢ e :: t, a∉FV(Γ)
$$\Gamma \vdash e :: t$$ ,
 \(a \not\in FV(\Gamma) \) ,
to $\Gamma \vdash e :: \forall a.t$
-->

For example

$${ { \vdash map :: (a\to b) \to [a] \to [b] } \over
   { \vdash map :: \forall b. (a\to b) \to [a] \to [b] } } \over
   { \vdash map :: \forall a. \forall b. (a\to b) \to [a] \to [b] } $$

Note:

$$ f : a \to b \not \vdash map\; f :: \forall b. [a] \to [b]  $$

* Instantiation

$$ {\Gamma \vdash e :: \forall a.t}\over {\Gamma \vdash e :: t[a:=s]} $$

# Classes

* Classes describe properties of types, e.g.

    ~~~~ {.haskell}
    class Eq a where
      (==) :: a -> a -> Bool
    instance Eq Bool where
       True  == True  = True
       False == False = True
       _     == _     = False

    class Eq a => Ord a where ...
    ~~~~

* types can be constrained by class context:

    ~~~~ {.haskell}
    elem :: Eq a => a -> [a] -> Bool
    ~~~~

+ Implementaction
    - an instance is translated to a method dictionary (akin to C++ vtable)
    - context is translated to an implicit parameter (method dictionary)
    - a subclass is translated to a function on method dicts


# Operations on types

* A simple example:

    ~~~~ {.haskell}
    data Tree a = Leaf | Node a (Tree a) (Tree a)
    ~~~~

* Type constructors transform types

* e.g. `Tree` maps `Int` to `Tree Int`

+ Higher order functions transform functions

+ Higher order constructors transform type constructors, e.g.

~~~~ {.haskell}
newtype IdentityT m a = IdentityT { runIdentityT :: m a }
~~~~

# Constructor classes

* constructor classes describe properties of type constructors:

    ~~~~ {.haskell}
    class Functor f where
      fmap :: (a->b) -> f a -> f b
    (<$>) = fmap

    instance Functor [] where
      fmap = map

    class Functor f => Pointed f where
       pure :: a -> f a
    instance Pointed [] where
       pure = (:[])

    class Pointed f => Applicative f where
      (<*>) :: f(a->b) -> f a -> f b

    instance Applicative [] where
      fs <*> xs = concat $ flip map fs (flip map xs)

    class Applicative m => Monad' m where
      (>>=) :: m a -> (a -> m b) -> m b
    ~~~~

<!--

    class Pointed f => Applicative f where
      (<*>) :: f(a->b) -> f a -> f b
      (*>) :: f a -> f b -> f b
      x *> y = (flip const) <$> x <*> y
      (<*) :: f a -> f b -> f a
      x <* y = const <$> x <*> y

    liftA2 :: Applicative f => (a -> b -> c) -> f a -> f b -> f c
    liftA2 f a b = f <$> a <*> b

-->

# Kinds

* Value operations are described by their types

* Type operations are described by their kinds

* Types (e.g.. `Int`, `Int -> Bool`) are of kind `*`

* One argument constructors are of type  (e.g.. `Tree`) are of kind `* -> *`

    ~~~~ {.haskell}
    {-#LANGUAGE KindSignatures, ExplicitForAll #-}

    class Functor f => Pointed (f :: * -> *) where
        pure :: forall (a :: *).a -> f a
    ~~~~

* More complex kinds are possible, e.g. for monad transformers:

    ~~~~ {.haskell}
    class MonadTrans (t :: (* -> *) -> * -> *) where
        lift :: Monad (m :: * -> *) => forall (a :: *).m a -> t m a
    ~~~~

NB spaces are obligatory - `::*->*` is one lexem

Newer Haskell versions allow introducing user kinds - we'll talk about them later.

# Multiparameter typeclasses

* Sometimes we need to describe a relationship between types rather than just a single type:

    ~~~~ {.haskell}
    {-#LANGUAGE MultiParamTypeClasses, FlexibleInstances #-}
    class Iso a b where
      iso :: a -> b
      osi :: b -> a

    instance Iso a a where
      iso = id
      osi = id

    instance Iso ((a,b)->c) (a->b->c) where
      iso = curry
      osi = uncurry

    instance (Iso a b) => Iso [a] [b] where
     iso = map iso
     osi = map osi
    ~~~~

* NB: in the last example `iso` has a different type on the left than on the right.

* Exercise: write more instances of `Iso`, e.g.


    ~~~~ {.haskell}
    instance (Functor f, Iso a b) => Iso (f a) (f b) where
    instance Iso (a->b->c) (b->a->c) where
    ~~~~

# Digression - FlexibleInstances

Haskell 2010

<!--
An instance declaration introduces an instance of a class. Let class
cx => C u where { cbody } be a class declaration. The general form of
the corresponding instance declaration is: instance cx′ => C (T u1 …
uk) where { d } where k ≥ 0. The type (T u1 … uk) must take the form
of a type constructor T applied to simple type variables u1, … uk;
furthermore, T must not be a type synonym, and the ui must all be
distinct.
-->

* an instance head must have the form `C (T u1 ... uk)`, where `T` is a type constructor defined by a data or newtype declaration  and the `u_i` are distinct type variables

<!--
*    and each assertion in the context must have the form C' v, where v is one of the ui.
-->

This prohibits instance declarations such as:

```
  instance C (a,a) where ...
  instance C (Int,a) where ...
  instance C [[a]] where ...
```

`instance Iso a a` does not meet these conditions, but it's easy to see  what relation we mean.

# Problem with muliparameter type classes (1)

Consider a class of collections, e.g.

`BadCollection.hs`

~~~~ {.haskell}
class Collection c where
  insert :: e -> c -> c
  member :: e -> c -> Bool

instance Collection [a] where
     insert = (:)
     member = elem
~~~~

we get an error:

~~~~
    Couldn't match type `e' with `a'
      `e' is a rigid type variable bound by
          the type signature for member :: e -> [a] -> Bool
          at BadCollection.hs:7:6
      `a' is a rigid type variable bound by
          the instance declaration
          at BadCollection.hs:5:22
~~~~

Why?

# Problem with muliparameter type classes (2)

~~~~ {.haskell}
class Collection c where
 insert :: e -> c -> c
 member :: e -> c -> Bool
~~~~

translates more or less to

~~~~
data ColDic c = CD
 {
 , insert :: forall e.e -> c -> c
 , member :: forall e.e -> c -> Bool
 }
~~~~

 ... this is not what we meant

~~~~ {.haskell}
instance Collection [a] where
   insert = (:)
   member = undefined
~~~~

~~~~
-- (:) :: forall t. t -> [t] -> [t]
ColList :: forall a. ColDic a
ColList = \@ a -> CD { insert = (:) @ a, member = undefined }
~~~~

# Problem with muliparameter type classes (3)

 <!--- `BadCollection2.hs` -->
<!---
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances #-}
-->

~~~~ {.haskell}
class Collection c e where
  empty :: c
  insert :: e -> c -> c
  member :: e -> c -> Bool

instance Eq a => Collection [a] a where
  empty = []
  insert  = (:)
  member = elem


ins2 x y c = insert y (insert x c)
-- ins2 :: (Collection c e, Collection c e1) => e1 -> e -> c -> c

problem1 :: [Int]
problem1 = ins2 1 2 []
-- No instances for (Collection [Int] e0, Collection [Int] e1)
-- arising from a use of `ins2'

problem2 = ins2 'a' 'b' []
-- No instance for (Collection [a0] Char)
--       arising from a use of `ins2'

problem3 :: (Collection c0 Char, Collection c0 Bool) => c0 -> c0
problem3 = ins2 True 'a'
-- Here the problem is that this is type correct, but shouldn't
~~~~


# Functional dependencies
Sometimes in multiparameter typeclasses, one parameter determines another, e.g.

~~~~ {.haskell}
 class (Monad m) => MonadState s m | m -> s where ...

 class Collects e ce | ce -> e where
      empty  :: ce
      insert :: e -> ce -> ce
      member :: e -> ce -> Bool
~~~~

Exercise: verify that `Collects` solves the problem we had with `Collection`


Problem: *Fundeps are very, very tricky.* - SPJ

More: https://www.microsoft.com/en-us/research/publication/understanding-functional-dependencies-via-constraint-handling-rules/ 

# Reflection - why not constructor classes?

We could try to solve the problem this way:

~~~~ {.haskell}
class Collection c where
  insert :: e -> c e -> c e
  member :: Eq e => e -> c e-> Bool

instance Collection [] where
     insert x xs = x:xs
     member = elem
~~~~

but this does not allow to solve the problem with the state monad:

~~~~ {.haskell}
 class (Monad m) => MonadState s m | m -> s where
   get :: m s
   put :: s -> m ()
~~~~

the state type `s` is not a parameter of `m`

# Fundeps are very very tricky

~~~~ {.haskell}
class Mul a b c | a b -> c where
  (*) :: a -> b -> c

newtype Vec a = Vec [a]
instance Functor Vec where
  fmap f (Vec as) = Vec $ map f as

instance Mul a b c => Mul a (Vec b) (Vec c) where
  a * b = fmap (a*) b

f t x y = if t then  x * (Vec [y]) else y
~~~~

What is the type of `f`? Let `x::a`, `y::b`.

Then the result type of `f` is `b` and we need an instance of `Mul a (Vec b) b`

Now  `a b -> c` implies `b ~ Vec c` for some `c`, so we are looking for an instance

~~~~
Mul a (Vec (Vec c)) (Vec c)
~~~~

applying the rule `Mul a b c => Mul a (Vec b) (Vec c)` leads to `Mul a (Vec c) c`.

...and so on


# Let's try

~~~~ {.haskell}
Mul1.hs:19:22: error:
    • Reduction stack overflow; size = 201
      When simplifying the following type: Mul a0 (Vec (Vec c)) (Vec c)
    • In the expression: x * (Vec [y])
      In the expression: if b then x * (Vec [y]) else y
      In an equation for ‘f’: f b x y = if b then x * (Vec [y]) else y
   |
19 | f b x y = if b then  x * (Vec [y]) else y
   |                      ^^^^^^^^^^^^^
~~~~

(we need to use UndecidableInstances, to make GHC try - this example shows what is 'Undecidable').

# Associated types

``` haskell
{-# LANGUAGE TypeFamilies #-}
class Collection c where
      type Elem c :: *
      empty :: c
      insert :: Elem c -> c -> c
      member :: Elem c -> c -> Bool

instance Eq a => Collection [a] where
  type Elem [a] = a
  empty = []
  insert  = (:)
  member = elem

ins2 :: Collection c => Elem c -> Elem c -> c -> c
ins2 x y c = insert y (insert x c)

noproblem1 :: [Int]
noproblem1 = ins2 (1::Int) (2::Int) empty

noproblem2 :: [Char]
noproblem2 = ins2 'a' 'b' empty
```

# Type families

Type families are functions on types

~~~~ {.haskell}
{-# LANGUAGE TypeFamilies #-}

data Zero = Zero
data Suc n = Suc n

type family m :+ n
type instance Zero :+ n = n
type instance (Suc m) :+ n = Suc(m:+n)

vhead :: Vec (Suc n) a -> a
vappend :: Vec m a -> Vec n a -> Vec (m:+n) a
~~~~

We'll talk about them systematically when we talk about dependent types in Haskell
