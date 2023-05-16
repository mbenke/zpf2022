{-# LANGUAGE TemplateHaskell #-}
import Language.Haskell.TH

main = putStrLn $(do{
                    s <- runIO (putStrLn "Enter string:" >> getLine);
                    stringE s})
