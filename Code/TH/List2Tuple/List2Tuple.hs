{-# LANGUAGE TemplateHaskell #-}
module List2Tuple where
import Language.Haskell.TH

tuple :: Int -> ExpQ
tuple n = [|\list -> $(tupE (exprs [|list|])) |]
  where
    exprs list = [infixE (Just (list))
                         (varE (mkName "!!"))
                         (Just (litE $ integerL (toInteger num)))
                    | num <- [0..(n - 1)]]