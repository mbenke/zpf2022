{-# LANGUAGE TemplateHaskell #-}

module Main where
import Derive

data Person = Person
  { personName :: String
  , personAge  :: Int
  , personCity :: String
  } deriving (Show)

-- This splice calls the macro at compile time.
-- reify ''Person sees the declaration above and generates:
--
--   fieldsOfPerson :: Person -> [(String, String)]
--   fieldsOfPerson x = [ ("personName", show (personName x))
--                       , ("personAge",  show (personAge  x))
--                       , ("personCity", show (personCity x)) ]
--
makeFieldsOf ''Person

main :: IO ()
main = do
  let p = Person "Alice" 30 "Warsaw"
  mapM_ (\(k,v) -> putStrLn (k ++ ": " ++ v)) (fieldsOfPerson p)
