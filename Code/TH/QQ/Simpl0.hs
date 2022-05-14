{-# LANGUAGE QuasiQuotes #-}
import Expr
import ExprQuoteManual

simpl :: Expr -> Expr
simpl (EAdd (EInt 0) x) = x
simpl e = e

main = print $ simpl [expr|0+4|]

