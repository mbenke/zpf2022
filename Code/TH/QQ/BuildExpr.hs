module BuildExpr where
import Language.Haskell.TH.Syntax

import Expr

mkInt :: Integer -> Exp
mkInt = AppE (ConE(mkName "EInt")) . LitE . IntegerL

mkBin :: String -> Exp -> Exp -> Exp
-- mkBin s e1 e2 = AppE (AppE (ConE (mkName s)) e1) e2
-- mkBin s = AppE . (AppE . ConE . mkName $ s)
mkBin = (AppE .) .(AppE . ConE . mkName)

mkAdd :: Exp -> Exp -> Exp
mkAdd = mkBin "EAdd"


mkIntP :: Integer -> Pat
mkIntP i = ConP (mkName "EInt") [LitP $ IntegerL i]

mkBinP :: String -> Pat -> Pat -> Pat
mkBinP s p1 p2 = ConP (mkName s) [p1, p2]

mkAddP :: Pat -> Pat -> Pat
mkAddP = mkBinP "EAdd"

instance Lift Expr where
