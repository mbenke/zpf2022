{-# LANGUAGE TemplateHaskell #-}
module Main where
import Language.Haskell.TH

import Build2

$(build_p1)

pprLn :: Ppr a => a -> IO ()
pprLn = putStrLn . pprint

main = do
  build_p1 >>= putStrLn . pprint
  print $ p1(1,2)

