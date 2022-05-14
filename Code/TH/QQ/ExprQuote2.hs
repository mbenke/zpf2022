-- Based on http://www.haskell.org/haskellwiki/Quasiquotation
module ExprQuote2 where

import Data.Generics.Aliases
import Language.Haskell.TH as TH
import Language.Haskell.TH.Quote

import qualified Expr2 as Expr

expr  :: QuasiQuoter
expr  =  QuasiQuoter
  { quoteExp = quoteExprExp
  , quotePat = quoteExprPat
  , quoteDec = undefined
  , quoteType = undefined
  }

quoteExprExp :: String -> Q Exp
quoteExprExp s = do
  pos <- getPosition
  exp <- Expr.parseExpr pos s
  dataToExpQ (const Nothing  `extQ` antiExprExp) exp

-- dataToExpQ :: Data a => (forall b. Data b => b -> Maybe (Q Exp)) -> a -> Q Exp
quoteExprPat :: String -> Q Pat
quoteExprPat s = do
  pos <- getPosition
  exp <- Expr.parseExpr pos s
  dataToPatQ (const Nothing `extQ` antiExprPat) exp

antiExprPat :: Expr.Expr -> Maybe (Q Pat)
antiExprPat (Expr.EMetaVar v) = Just $ varP (mkName v)
antiExprPat _ = Nothing

antiExprExp :: Expr.Expr -> Maybe (Q Exp)
antiExprExp (Expr.EMetaVar v) = Just $ varE (mkName v)
antiExprExp _ = Nothing

getPosition = fmap transPos location where
  transPos loc = (loc_filename loc,
                  fst (loc_start loc),
                  snd (loc_start loc))
