{-# LANGUAGE TemplateHaskell #-}
import List2Tuple
main = do
  print ($(tuple 8) [1..8])
  print ($(tuple 3) [1..])