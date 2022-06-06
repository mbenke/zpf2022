{-# LANGUAGE MultiParamTypeClasses, FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE DataKinds, KindSignatures #-}
{-# LANGUAGE TypeFamilies, TypeOperators #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE ExplicitForAll #-}

module ClassAdd where

data Nat :: * where
  Z :: Nat
  S :: Nat -> Nat

infixr 6 :>
data Vec :: Nat -> * -> * where
  V0   :: Vec 'Z a
  (:>) :: a -> Vec n a -> Vec ('S n) a

class Add (a::Nat) (b::Nat) (c::Nat)  where

instance Add  Z    b  b
instance Add a b c => Add (S a) b (S c)

vappend :: (Add m n s) => Vec m a -> Vec n a -> Vec s a
vappend = undefined
-- vappend V0 ys = ys
