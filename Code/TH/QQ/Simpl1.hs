{-# LANGUAGE QuasiQuotes #-}
import Expr
import ExprQuote1

simpl :: Expr -> Expr
simpl (EAdd (EInt 0) x) = x

main = print $ simpl [expr|0+2|]

