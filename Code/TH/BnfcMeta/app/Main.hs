{-# LANGUAGE QuasiQuotes, TemplateHaskell #-}
module Main where
import Language.LBNF.Runtime(printTree)
    
import Expr

exp1 :: Expr
exp1 = [expr| 2 + 2 |]

main :: IO ()
main = putStrLn (printTree exp1)
-- putStrLn "Hello, Haskell!"
