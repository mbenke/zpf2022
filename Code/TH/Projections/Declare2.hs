{-# LANGUAGE TemplateHaskell #-}
module Main where
import Language.Haskell.TH

import Build2

$(build_p1)

pprLn :: Ppr a => a -> IO ()
pprLn = putStrLn . pprint
main = do
  decs <- runQ build_p1
  pprLn decs
  print $ p1(1,2)

