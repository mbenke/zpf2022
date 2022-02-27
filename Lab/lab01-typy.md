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

Try to write `vappend` and see what the problem is.


# Type classes

Fill missing definitions:

~~~~  {.haskell}
class Fluffy f where
  furry :: (a -> b) -> f a -> f b

-- Exercise 1
-- Relative Difficulty: 1
instance Fluffy [] where
  furry = error "todo"

-- Exercise 2
-- Relative Difficulty: 1
instance Fluffy Maybe where
  furry = error "todo"

-- Exercise 3
-- Relative Difficulty: 5
instance Fluffy ((->) t) where
  furry = error "todo"

newtype EitherLeft b a = EitherLeft (Either a b)
newtype EitherRight a b = EitherRight (Either a b)

-- Exercise 4
-- Relative Difficulty: 5
instance Fluffy (EitherLeft t) where
  furry = error "todo"

-- Exercise 5
-- Relative Difficulty: 5
instance Fluffy (EitherRight t) where
  furry = error "todo"

class Misty m where
  banana :: (a -> m b) -> m a -> m b
  unicorn :: a -> m a
  -- Exercise 6
  -- Relative Difficulty: 3
  -- (use banana and/or unicorn)
  furry' :: (a -> b) -> m a -> m b
  furry' = error "todo"

-- Exercise 7
-- Relative Difficulty: 2
instance Misty [] where
  banana = error "todo"
  unicorn = error "todo"

-- Exercise 8
-- Relative Difficulty: 2
instance Misty Maybe where
  banana = error "todo"
  unicorn = error "todo"

-- Exercise 9
-- Relative Difficulty: 6
instance Misty ((->) t) where
  banana = error "todo"
  unicorn = error "todo"

-- Exercise 10
-- Relative Difficulty: 6
instance Misty (EitherLeft t) where
  banana = error "todo"
  unicorn = error "todo"

-- Exercise 11
-- Relative Difficulty: 6
instance Misty (EitherRight t) where
  banana = error "todo"
  unicorn = error "todo"

-- Exercise 12
-- Relative Difficulty: 3
jellybean :: (Misty m) => m (m a) -> m a
jellybean = error "todo"

-- Exercise 13
-- Relative Difficulty: 6
apple :: (Misty m) => m a -> m (a -> b) -> m b
apple = error "todo"

-- Exercise 14
-- Relative Difficulty: 6
moppy :: (Misty m) => [a] -> (a -> m b) -> m [b]
moppy = error "todo"

-- Exercise 15
-- Relative Difficulty: 6
-- (bonus: use moppy)
sausage :: (Misty m) => [m a] -> m [a]
sausage = error "todo"

-- Exercise 16
-- Relative Difficulty: 6
-- (bonus: use apple + furry')
banana2 :: (Misty m) => (a -> b -> c) -> m a -> m b -> m c
banana2 = error "todo"

-- Exercise 17
-- Relative Difficulty: 6
-- (bonus: use apple + banana2)
banana3 :: (Misty m) => (a -> b -> c -> d) -> m a -> m b -> m c -> m d
banana3 = error "todo"

-- Exercise 18
-- Relative Difficulty: 6
-- (bonus: use apple + banana3)
banana4 :: (Misty m) => (a -> b -> c -> d -> e) -> m a -> m b -> m c -> m d -> m e
banana4 = error "todo"

newtype State s a = State {
  state :: (s -> (s, a))
}

-- Exercise 19
-- Relative Difficulty: 9
instance Fluffy (State s) where
  furry = error "todo"

-- Exercise 20
-- Relative Difficulty: 10
instance Misty (State s) where
  banana = error "todo"
  unicorn = error "todo"
~~~~

source:

http://blog.tmorris.net/posts/20-intermediate-haskell-exercises/

# Multiparam type classes

Consider class Iso from the lecture

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


* Exercise: write more instances of `Iso`, e.g.

    ~~~~ {.haskell}
    instance (Functor f, Iso a b) => Iso (f a) (f b) where ...
    instance Iso (a->b->c) (b->a->c) where ...
    instance (Monad m, Iso a b) => Iso (m a) (m b) where ...
    ~~~~

# Fundeps are very very tricky

Define

~~~~ {.haskell}
{-# LANGUAGE FlexibleInstances, FlexibleContexts, UndecidableInstances #-}
{-# LANGUAGE MultiParamTypeClasses, FunctionalDependencies #-}

class Mul a b c | a b -> c where
  (*) :: a -> b -> c

newtype Vec a = Vec [a]
instance Functor Vec where
  fmap f (Vec as) = Vec $ map f as

instance Mul a b c => Mul a (Vec b) (Vec c) where
  a * b = fmap (a*) b

f t x y = if t then  x * (Vec [y]) else y
~~~~

and watch it explode ;)
