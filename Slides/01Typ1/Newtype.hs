{-# LANGUAGE TypeFamilies #-}
module Newtype where

class Newtype new where
  type Old new
  pack :: Old new -> new
  unpack :: new -> Old new

newly :: (Newtype a, Newtype b) => (Old a -> Old b) -> a -> b 
newly f = pack . f . unpack

-- we might use type application instead of the dummy arg          
ala :: (Newtype b, Newtype d) => ((a -> b    ) -> c ->     d) -> (Old b -> b)
                              ->  (a -> Old b) -> c -> Old d
ala hof _ f = unpack . hof (pack .f)
  
-- traverse `ala` I [0..5]
-- traverse `ala` K `ala` Sum (+1) [0..5]
