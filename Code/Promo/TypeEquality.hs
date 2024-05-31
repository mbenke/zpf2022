{-# LANGUAGE GADTs #-}
{-# LANGUAGE DataKinds, KindSignatures, PolyKinds #-}
{-# LANGUAGE TypeFamilies, TypeOperators #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE ExplicitForAll #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}

import Data.Kind

-- cf Data.Type.Equality

infix 4 :~:

data a :~: b where
  Refl ::  a :~: a

sym :: (a :~: b) -> (b :~: a)
sym Refl = Refl  -- seems trivial, but see if you can simplify it...

trans :: (a :~: b) -> (b :~: c) -> (a :~: c)
trans Refl Refl = Refl

cong :: forall f a b.a :~: b -> f a :~: f b
cong Refl = Refl

-- (a ~ b) implies (f a) implies (f b)
subst :: a :~: b -> f a -> f b
subst Refl = id

-- | Typesafe cast using propositional equality
castWith :: (a :~: b) -> a -> b
castWith Refl x = x

-- | Generalised form of typesafe cast
gcastWith :: (a :~: b) -> (a ~ b => r) -> r
gcastWith Refl x = x

-- # Nat
data Nat = Z | S Nat

infixl 6 :+
type family (n::Nat) :+ (m::Nat) :: Nat where
  'Z :+ m = m
  'S n :+ m = 'S(n :+ m)

data SNat n where
  SZ :: SNat 'Z
  SS :: SNat n -> SNat ('S n)

-- # Nat Proofs
-- this is trivial
plus_id_l :: SNat n -> 'Z :+ n :~: n
plus_id_l _ = Refl


-- implicit quant
-- plus_id_l_impl :: forall (n::Nat).(Z :+ n) :~: n 
plus_id_l_impl :: ('Z :+ n) :~: n 
plus_id_l_impl = Refl

plus_id_r :: SNat n -> n :+ 'Z :~: n
plus_id_r SZ = Refl
plus_id_r (SS m) = cong @'S (plus_id_r m)
-- @S is optional above, added only for clarity

-- S m + Z ~ S(m + Z)
succ_plus_id :: SNat n1 -> SNat n2 -> (S n1) :+ n2 :~: S (n1 :+ n2)
succ_plus_id _ _ = Refl

-- succ_plus_id_impl :: (S n1) :+ n2 :~: S (n1 :+ n2)
-- succ_plus_id_impl = Refl

infixr 6 :>
data Vec :: Type -> Nat ->Type where
  V0 :: Vec a Z
  (:>) :: a -> Vec a n -> Vec a (S n)

deriving instance Show a => Show (Vec a n)

toList :: Vec a n -> [a]
toList V0 = []
toList (x:>xs) = x:toList xs

size :: Vec a n -> SNat n
size V0 = SZ
size (_:>t) = SS (size t)

simpl0r :: SNat n -> f (n:+Z) -> f n
-- Special case: Vec (n:+Z) a -> Vec n a
simpl0r n v = subst (plus_id_r n) v

expand0r :: SNat n -> f n -> f(n:+Z)
expand0r n x = subst (sym (plus_id_r n)) x

-- Instead of `subst ... sym` we can put constraint solver to work
-- you can think of it as a kind of tactic
expand0r' :: SNat n -> f n -> f(n:+Z)
expand0r' n x = gcastWith (plus_id_r n) x

plus_succ_r :: forall m n. SNat n -> n :+ S m :~: S(n :+ m)
plus_succ_r SZ = Refl
plus_succ_r (SS n1) = cong @S (plus_succ_r @m n1)

plus_succ_r2 :: SNat n -> SNat m -> n :+ S m :~: S(n :+ m)
plus_succ_r2 SZ m = Refl
plus_succ_r2 (SS n1) m = cong @S (plus_succ_r2 n1 m)

rev :: [a] -> [a]
rev [] = []
rev xs = go [] xs where
  go acc [] = acc
  go acc (h:t) = go (h:acc) t

accrev :: Vec a n -> Vec a n
-- accrev V0 = V0
accrev xs = go SZ V0 xs where
  by :: (x ~ y => r) -> (x :~: y) -> r
  value `by` proof = gcastWith proof value
  go :: forall m n a.SNat n -> Vec a n -> Vec a m -> Vec a (n:+m)
  go alen acc V0     = acc
                       `by` plus_id_r alen               -- x + Z ~ x
  go alen acc (h:>t) =  go (SS alen) (h:>acc) t 
                       `by` (plus_succ_r2 alen (size t)) -- x + S y ~ S(x+y)         

main :: IO ()
main = print $ accrev $ 1 :> 2 :> 3 :> V0

--Exercise: implement a vector variant of
naiverev :: [a] -> [a]
naiverev [] = []
naiverev (x:xs) = naiverev xs ++ [x]

-- Exercise: try to eliminate `size` from `accrev` by using proxies or type app
