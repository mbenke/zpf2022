{-# LANGUAGE TemplateHaskell #-}
module Build2 where
import Language.Haskell.TH

build_p1 :: Quote m => m [Dec]
build_p1 = do
  let p1 = mkName "p1"  
  a <- newName "a"
  b <- newName "a"
  return
    [ FunD p1 
             [ Clause [TupP [VarP a,VarP b]] (NormalB (VarE a)) []
             ]
    ]
