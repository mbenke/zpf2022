{-# LANGUAGE QuasiQuotes, TemplateHaskell #-}
{-# OPTIONS_GHC -Wno-unused-local-binds -Wno-unused-matches -Wno-missing-signatures #-}
{-# OPTIONS_GHC -Wno-name-shadowing #-}
module Expr where
import Language.LBNF.Compiletime
import Language.LBNF(lbnf, bnfc)

bnfc [lbnf|
EAdd . Expr1 ::= Expr1 "+" Expr2 ;
EMul . Expr2 ::= Expr2 "*" Expr3 ;
ELit . Expr3 ::= Integer ;
EVar . Expr3 ::= Ident ;
coercions Expr 3;
|]
