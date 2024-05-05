#!/usr/bin/env cabal
{- cabal:
build-depends:
  base ^>=4.16.0.0,
  random ^>=1.2,
-}

import System.Random
  ( StdGen       -- :: *
  , newStdGen    -- :: IO StdGen
  , randomR      -- :: (RandomGen g, Random a) => (a, a) -> g -> (a, g)
  , split        -- :: RandomGen g => g -> (g, g)
                 -- splits its argument into independent generators
  -- instance RandomGen StdGen
  -- instance Random Int
  )

roll :: StdGen -> Int
roll rnd = fst $ randomR (1,6) rnd
main = do
  (r1,r2) <- fmap split newStdGen
  print (roll r1)
  print (roll r2)
  print (roll r1)
  print (roll r2)
