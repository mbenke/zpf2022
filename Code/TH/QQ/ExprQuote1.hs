-- Based on http://www.haskell.org/haskellwiki/Quasiquotation
module ExprQuote1 where

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

quoteExprExp :: String -> Q Exp
quoteExprExp s = do
  pos <- getPosition
  exp <- parseExpr pos s

  dataToExpQ (const Nothing) exp -- equivalently `liftData exp`
-- dataToExpQ :: Data a => (forall b. Data b => b -> Maybe (Q Exp)) -> a -> Q Exp

quoteExprPat :: String -> Q Pat
quoteExprPat s = do
  pos <- getPosition
  exp <- parseExpr pos s
  dataToPatQ (const Nothing) exp

getPosition = fmap transPos location where
  transPos loc = (loc_filename loc,
                  fst (loc_start loc),
                  snd (loc_start loc))
