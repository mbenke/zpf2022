{-# LANGUAGE TemplateHaskell #-}
import Language.Haskell.TH

power :: Int -> Q Exp
power 0 = [| const 1 |]
power n = [| \k -> k * $(power (n-1)) k |]

{-
λ> :set -XTemplateHaskell
λ> $(power 5) 2
32
-}
