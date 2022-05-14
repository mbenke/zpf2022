 -- {- # LANGUAGE TemplateHaskell #-}
module Build3 where
import Language.Haskell.TH

simpleFun :: Name -> [Pat] -> Exp -> Dec
simpleFun name pats rhs = FunD name [Clause pats (NormalB rhs) []]

build_ps = mapM build_p [1,2] where
    fname n = mkName $ "p2_" ++ show n
    build_p n = do
        argNames <- mapM newName (replicate 2 "a")
        let args = map VarP argNames
        return $ simpleFun (fname n) [TupP args] (VarE (argNames !! (n-1)))
