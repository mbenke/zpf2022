{-# LANGUAGE MultiParamTypeClasses, FunctionalDependencies,
   FlexibleInstances, FlexibleContexts, UndecidableInstances, MonoLocalBinds #-}
-- static Peano constructors and numerals

data Zero
data Succ n

type One   = Succ Zero
type Two   = Succ One
type Three = Succ Two
type Four  = Succ Three


-- dynamic representatives for static Peanos

zero  = undefined :: Zero
one   = undefined :: One
two   = undefined :: Two
three = undefined :: Three
four  = undefined :: Four


-- addition, a la Prolog

class Add a b c | a b -> c where
  add :: a -> b -> c
  add = undefined
instance              Add  Zero    b  b
instance Add a b c => Add (Succ a) b (Succ c)


-- multiplication, a la Prolog

class Mul a b c | a b -> c where
  mul :: a -> b -> c
  mul = undefined

instance                           Mul  Zero    b Zero
instance (Mul a b c, Add b c d) => Mul (Succ a) b d


-- factorial, a la Prolog

class Fac a b | a -> b where
  fac :: a -> b
  fac = undefined

instance                                Fac  Zero    One
instance (Fac n k, Mul (Succ n) k m) => Fac (Succ n) m

-- try, for "instance" (sorry):
--
--     :t fac four
