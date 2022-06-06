{-# LANGUAGE GADTs #-}
{-# LANGUAGE DataKinds, KindSignatures, PolyKinds #-}
{-# LANGUAGE TypeFamilies, TypeOperators #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE ExplicitForAll #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE StandaloneKindSignatures #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE TemplateHaskell #-}

module Singletons where
import Data.Singletons
import Data.Singletons.TH


$(singletons [d|
    data Nat :: * where
      Z :: Nat
      S :: Nat -> Nat

    plus :: Nat -> Nat -> Nat
    plus Z     m = m
    plus (S n) m = S (plus n m)
    |])

  
infixr 6 :>
data Vec :: Nat -> * -> * where
  V0   :: Vec 'Z a
  (:>) :: a -> Vec n a -> Vec ('S n) a

deriving instance (Show a) => Show (Vec n a)

vchop :: Sing m -> Vec(Plus m n) a ->  (Vec m a, Vec n a)
vchop SZ xs = (V0, xs)
vchop (SS m) (x:>xs) = (x:>ys, zs) where
  (ys,zs) = vchop m xs
