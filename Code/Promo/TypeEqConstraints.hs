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

-- cf Data.Type.Equality

infix 4 :~:

-- data a :~: b where
--   Refl :: (a :~: a)
  
data a :~: b = (a ~ b) => Refl

sym :: (a :~: b) ->  (b ~ a => r) -> r
sym Refl x = x

-- | Generalised form of typesafe cast
gcastWith :: (a :~: b) -> (a ~ b => r) -> r
gcastWith Refl x = x

trans :: (a :~: b) -> (b :~: c) -> (a :~: c)
trans Refl Refl = Refl

cong :: forall f a b.a :~: b -> f a :~: f b
cong Refl = Refl
