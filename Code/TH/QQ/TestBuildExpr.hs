{-# LANGUAGE TemplateHaskell #-}
import Language.Haskell.TH.Syntax
import BuildExpr
import Expr

testExpr :: Expr
testExpr = $(return $ mkAdd (mkInt 0) (mkInt 2))

simpl :: Expr -> Expr
simpl $(return $ mkAddP (mkIntP 0) (VarP (mkName "x"))) = x
simpl e = e
