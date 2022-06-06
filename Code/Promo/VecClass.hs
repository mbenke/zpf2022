{-# LANGUAGE MultiParamTypeClasses, FunctionalDependencies, FlexibleInstances, FlexibleContexts, UndecidableInstances #-}
{-# LANGUAGE TypeFamilies, GADTs, EmptyDataDecls #-}
{-# LANGUAGE ScopedTypeVariables #-}
-- > {-# LANGUAGE KindSignatures, TypeOperators, FlexibleContexts #-}

-- static Peano constructors and numerals

data Zero
data Succ n

type One   = Succ Zero
type Two   = Succ One
type Three = Succ Two
type Four  = Succ Three

class Nat n where
instance Nat Zero
instance Nat n => Nat (Succ n)  

-- dynamic representatives for static Peanos

zero  = undefined :: Zero
one   = undefined :: One
two   = undefined :: Two
three = undefined :: Three
four  = undefined :: Four


-- addition, a la Prolog

-- class (Nat a, Nat b) => Add a b c | a b -> c where
class (Nat a, Nat b) => Add a b c | a b -> c where
  add :: a -> b -> c
  add = undefined
instance Nat b =>     Add  Zero    b  b
instance Add a b c => Add (Succ a) b (Succ c)


infixr 5 :>
data Vec :: * -> * -> * where
  VNil :: Vec Zero a  
  (:>) :: a -> Vec n a -> Vec (Succ n) a

vhead :: Vec (Succ n) a -> a
vhead (x :> xs) = x

vtail :: Vec (Succ n) a -> Vec n a
vtail (x :> xs) = xs

vlast :: Vec (Succ n) a -> a
vlast (x :> VNil) = x
vlast (x :> xs@(y :> ys)) = vlast xs


vappend :: (Nat m, Nat n, Add m n s) => Vec m a -> Vec n a -> Vec s a
vappend xs@VNil ys = ys
vappend xs ys = undefined


