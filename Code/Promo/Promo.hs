{-# LANGUAGE GADTs #-}
{-# LANGUAGE DataKinds, KindSignatures, PolyKinds #-}
{-# LANGUAGE TypeFamilies, TypeOperators #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE ExplicitForAll #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE RankNTypes #-}

module Promo where

data Nat :: * where
  Z :: Nat
  S :: Nat -> Nat

-- This defines
-- Type Nat
-- Value constructors: Z, S

-- Promotion (lifting) to type level yields
-- kind Nat
-- type constructors: 'Z :: Nat; 'S :: Nat -> Nat
-- 's can be omitted in most cases, but...

-- data P          -- 1
-- data Prom = P   -- 2
-- type T = P      -- 1 or promoted 2?
-- quote disambiguates:
-- type T1 = P     -- 1
-- type T2 = 'P    -- promoted 2

-- Other promotions
data HList :: [*] -> * where
  HNil  :: HList '[]
  HCons :: a -> HList t -> HList (a ': t)

data Tuple :: (*,*) -> * where
  Tuple :: a -> b -> Tuple '(a,b)

foo0 :: HList '[]
foo0 = HNil

foo1 :: HList '[Int]
foo1 = HCons 3 HNil

foo2 :: HList [Int, Bool]
foo2 = HCons 3 (HCons True HNil)

infixr 6 :>
data Vec :: Nat -> * -> * where
  V0   :: Vec 'Z a
  (:>) :: a -> Vec n a -> Vec ('S n) a

deriving instance (Show a) => Show (Vec n a)


infixl 6 :+

type family (n :: Nat) :+ (m :: Nat) :: Nat
type instance Z :+ m = m
type instance (S n) :+ m = S (n :+ m)

vhead :: Vec ('S n) a -> a
vhead (x:>_) = x

vtail :: Vec ('S n) a -> Vec n a
vtail (_:> xs) = xs

vapp :: Vec m a -> Vec n a -> Vec (m :+ n) a
vapp V0 ys = ys
vapp (x:>xs) ys = x:>(vapp xs ys)

-- |
-- Indexing
-- >>> (1:>V0) `atIndex` FinZ
-- 1
--
-- atIndex :: Vec n a -> (m < n) -> a

data Fin n where
    FinZ :: Fin ('S n) -- zero is less than any successor
    FinS :: Fin n -> Fin ('S n) -- n is less than (n+1)

atIndex :: Vec n a -> Fin n -> a
atIndex (x:>_) FinZ = x
atIndex (_:>xs) (FinS k) = atIndex xs k

-- Exercise: why not:
-- atIndex :: Vec (S n) a -> ... ?
-- atIndex2 :: Vec (S n) a -> Fin (S n) -> a
-- atIndex2 (x:>_) FinZ = x
-- atIndex2 (_:>xs) (FinS k) = atIndex2 xs k

-- Want
vreplicate1 :: Nat -> a -> Vec n a
vreplicate1 = undefined
-- vreplicate Z _ = V0   --  Expected type: Vec n a
                         --  Actual type:   Vec 'Z a

-- this does not work either
-- vreplicate2 :: (n::Nat) -> a -> Vec n a


vchop1 :: Vec (m :+ n) a -> (Vec m a, Vec n a)
vchop1 _ = undefined

-- | Chop a vector in two, using first argument as a measure
-- >>> vchop2 (() :> V0) (1 :> 2 :> V0)
-- (1 :> V0,2 :> V0)

-- NB if we had `vreplicate`, we might write
-- vchop2 (vreplicate (S Z) ()) (1 :> 2 :> V0)
vchop2 :: Vec m x -> Vec (m :+ n) a -> (Vec m a, Vec n a)
vchop2 V0 xs = (V0, xs)
vchop2 (_:>m) (x:>xs) = (x:>ys, zs) where
  (ys, zs) = vchop2 m xs

-- inhabitants of Nat types
data SNat n where
  SZ :: SNat 'Z
  SS :: SNat n -> SNat ('S n)
deriving instance Show(SNat n)

add :: (SNat m) -> (SNat n) -> SNat(m :+ n)
add SZ n = n
add (SS m) n = SS (add m n)

-- | `vreplicate n a` is a vector of n copies of a
-- >>> vreplicate (SS SZ) 1
-- 1 :> V0
-- >>> vreplicate (SS (SS SZ)) 1
-- 1 :> (1 :> V0)
vreplicate :: SNat n -> a -> Vec n a
vreplicate SZ _ = V0
vreplicate (SS n) x = x:>(vreplicate n x)

-- | chop a vector in two parts
-- >>> vchop (SS SZ) (1 :> 2 :> V0)
-- (1 :> V0,2 :> V0)
vchop :: SNat m -> Vec(m:+n) a -> (Vec m a, Vec n a)
vchop = vchop3
vchop3 :: SNat m -> Vec(m:+n) a -> (Vec m a, Vec n a)
vchop3 SZ xs = (V0, xs)
vchop3 (SS m) (x:>xs) = (x:>ys, zs) where
  (ys,zs) = vchop3 m xs

-- Exercise: define multiplication

-- # Comparison

type family (m::Nat) :< (n::Nat) :: Bool
type instance m :< 'Z = 'False
type instance 'Z :< ('S n) = 'True
type instance ('S m) :< ('S n) = m :< n

-- nth
nth :: (m:<n) ~ 'True => SNat m -> Vec n a -> a
nth SZ (a:>_)  = a
nth (SS m') (_:>xs) = nth m' xs

-- >>> vtake0 (SS SZ) (1 :> 2 :> V0)
-- 1 :> V0
vtake0 :: forall m n a.SNat m -> Vec (m :+ n) a -> Vec m a
vtake0 m v = let (a,b) = vchop m v :: (Vec m a, Vec n a) in a
-- vchop :: SNat m -> Vec(m:+n) a -> (Vec m a, Vec n a)

-- | Take first `n` elements of a vector
vtake1 :: SNat m -> Vec (m :+ n) a -> Vec m a
vtake1  SZ    _ = V0
-- vtake1 (SS m) (x:>xs) = x :> vtake1 m xs
vtake1  _ _ = undefined

-- | vtake1' - concrete handles
-- >>> let v = 1 :> (1 :> (1 :> V0)); two = SS(SS SZ) in vtake1' two (SS SZ) v
-- 1 :> (1 :> V0)
vtake1' :: SNat m -> SNat n -> Vec (m :+ n) a -> Vec m a
vtake1' SZ _  _ = V0
vtake1' (SS m) n (x:>xs) = x :> vtake1' m n xs

-- | Nat Proxy
data NP :: Nat -> * where NP :: NP n

-- >>> let v = 1 :> (1 :> (1 :> V0)); two = SS(SS SZ) in vtake2 two NP v
-- 1 :> (1 :> V0)
vtake2 :: SNat m -> NP n -> Vec (m :+ n) a -> Vec m a
vtake2 SZ     _ _ = V0
vtake2 (SS m) n (x:>xs) = x :> vtake2 m n xs

-- | Generic Proxy
data Proxy :: k -> * where
  Proxy :: Proxy (i::k)

-- >>> let v = 1 :> (1 :> (1 :> V0)); two = SS(SS SZ) in vtake3 two Proxy v
-- 1 :> (1 :> V0)
vtake3 :: SNat m -> Proxy n -> Vec (m :+ n) a -> Vec m a
vtake3 SZ     _ _ = V0
vtake3 (SS m) n (x:>xs) = x :> vtake3 m n xs


-- vtake4 requires:
-- {-# LANGUAGE ScopedTypeVariables #-}
-- {-# LANGUAGE TypeApplications #-} -- GHC>=8.0

-- >>> let v = 1 :> (1 :> (1 :> V0)); two = SS(SS SZ) in vtake4 two v
-- 1 :> (1 :> V0)
vtake4 :: forall n m a. SNat m -> Vec (m :+ n) a -> Vec m a
vtake4 SZ _ = V0
vtake4 (SS m) (x:>xs) = x :> vtake4 @n m xs

class SNAT(n::Nat) where
  snat :: SNat n

instance SNAT Z where
  snat = SZ

instance SNAT n => SNAT (S n) where
  snat = SS snat

-- >>> vtrunc Proxy (1 :> 2 :> 3 :> 4 :> V0) :: Vec (S (S Z)) Int
-- 1 :> (2 :> V0)
vtrunc :: SNAT m => Proxy n -> Vec (m :+ n) a -> Vec m a
vtrunc = vtake3 snat

-- >>> vtrunc2 (1 :> 2 :> 3 :> 4 :> V0) :: Vec (S (S Z)) Int
-- 1 :> (2 :> V0)
vtrunc2 :: forall n m a. SNAT m => Vec (m :+ n) a -> Vec m a
vtrunc2 = vtake4 @n @m (snat::SNat m)


naiverev :: [a] -> [a]
naiverev [] = []
naiverev (x:xs) = naiverev xs ++ [x]

vrev1 :: Vec n a -> Vec n a
vrev1 V0 = V0
-- vrev1 (x:>xs) = vapp (vrev1 xs) (x:>V0)
vrev1 (x:>xs) = undefined

-- | vrev2
-- >>> vrev2 (1:>2:>3:>V0)
-- 3 :> (2 :> (1 :> V0))
vrev2 :: Vec n a -> Vec n a
vrev2 V0 = V0
vrev2 (x:>xs) = snoc (vrev2 xs) x

snoc :: Vec n a -> a -> Vec (S n) a
snoc V0 y = y :> V0
snoc (x:>xs) y = x :> snoc xs y


rev :: [a] -> [a]
rev [] = []
rev xs = accrev [] xs

accrev :: [a] -> [a] -> [a]
accrev acc [] = acc
accrev acc (h:t) = accrev (h:acc) t


vrev3 :: Vec n a -> Vec n a
vrev3 xs = vaccrev V0 xs

vaccrev :: Vec n a -> Vec m a -> Vec (n :+ m) a
--vaccrev acc V0 = acc
vaccrev = undefined
