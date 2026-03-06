{-# LANGUAGE TemplateHaskell #-}
module Derive where
import Language.Haskell.TH

-- | makeToFields ''Person
-- generates:  fieldsOfPerson :: Person -> [(String, String)]
makeFieldsOf :: Name -> Q [Dec]
makeFieldsOf typeName = do
  info <- reify typeName
  fields <- case info of
    TyConI (DataD _ _ _ _ [RecC _ fs] _) -> return fs
    _ -> fail $ show typeName ++ " must be a single-constructor record"

  xName <- newName "x"

  -- derive "fieldsOfPerson" from "Person"
  let funName = mkName ("fieldsOf" ++ nameBase typeName)   -- nameBase :: Name -> String

  -- each field becomes ("fieldName", show (fieldAccessor x))
  let mkPair (fieldName, _, _) =
        TupE [ Just (LitE (StringL (nameBase fieldName)))
             , Just (AppE (VarE 'show) (AppE (VarE fieldName) (VarE xName)))
             ]

  let body = NormalB (ListE (map mkPair fields))
  return [ FunD funName [ Clause [VarP xName] body [] ] ]
