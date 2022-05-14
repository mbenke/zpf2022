-- Based on http://www.haskell.org/haskellwiki/Quasiquotation
module ExprQuoteManual where

-- import Data.Generics
import Language.Haskell.TH as TH
import Language.Haskell.TH.Quote
import Language.Haskell.TH.Syntax
import Expr

expr  :: QuasiQuoter
expr  =  QuasiQuoter
  { quoteExp = quoteExprExp
  , quotePat = quoteExprPat
  , quoteDec = undefined
  , quoteType = undefined
  }

-- ## Quote Expressions

quoteExprExp :: String -> Q Exp
quoteExprExp s = do
  pos <- getPosition
  exp <- parseExpr pos s
  exprToExpQ exp
  -- liftData exp

exprToExpQ :: Expr -> Q Exp
exprToExpQ (EInt n) = return $ ConE (mkName "EInt") $$ (intLitE n)
exprToExpQ (EAdd e1 e2) = convertBinE "EAdd" e1 e2
exprToExpQ (ESub e1 e2) = convertBinE "ESub" e1 e2
exprToExpQ (EMul e1 e2) = convertBinE "EMul" e1 e2
exprToExpQ (EDiv e1 e2) = convertBinE "EDiv" e1 e2

intLitE :: Int -> TH.Exp
intLitE = LitE . IntegerL . toInteger

convertBinE s e1 e2 = do
  e1' <- exprToExpQ e1
  e2' <- exprToExpQ e2  
  return $ ConE (mkName s) $$ e1' $$ e2'
  
infixl 1 $$
($$) = AppE

-- ## Quote Patterns

quoteExprPat :: String -> Q Pat
quoteExprPat s = do
  pos <- getPosition
  exp <- parseExpr pos s
  exprToPatQ exp
  -- dataToPatQ (const Nothing) exp

getPosition = fmap transPos location where
  transPos loc = (loc_filename loc,
                  fst (loc_start loc),
                  snd (loc_start loc))

exprToPatQ :: Expr -> Q Pat
exprToPatQ (EInt n) = return $ ConP (mkName "EInt") [intLitP n]
exprToPatQ (EAdd e1 e2) = convertBinP "EAdd" e1 e2
exprToPatQ (ESub e1 e2) = convertBinP "ESub" e1 e2
exprToPatQ (EMul e1 e2) = convertBinP "EMul" e1 e2
exprToPatQ (EDiv e1 e2) = convertBinP "EDiv" e1 e2

intLitP :: Int -> TH.Pat
intLitP = LitP . IntegerL . toInteger

convertBinP s e1 e2 = do
  e1' <- exprToPatQ e1
  e2' <- exprToPatQ e2  
  return $ ConP (mkName s) [e1', e2']
  
