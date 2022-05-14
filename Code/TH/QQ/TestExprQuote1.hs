{-# LANGUAGE  QuasiQuotes #-}
import ExprQuote1
import Expr

-- show
testExp :: Expr
testExp = [expr|1+2*3|]

f1 :: Expr -> String
f1 [expr| 1 + 2*3 |] = "Bingo!"
f1 _ = "Sorry, no bonus"

main = putStrLn $ f1 testExp
-- /show
