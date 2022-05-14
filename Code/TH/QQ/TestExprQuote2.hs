{-# LANGUAGE  QuasiQuotes #-}
import ExprQuote2
import Expr2

t1 :: Expr
t1 = [expr|1+2*3|]

f1 :: Expr -> String
f1 [expr| $a + $b |] = "Bingo!"
f1 _ = "Sorry, no bonus"

eval [expr| $a + $b|] = eval a + eval b
eval [expr| $a * $b|] = eval a * eval b
eval (EInt n) = n

test = eval [expr| 2+2 |]

twice :: Expr -> Expr
twice e = [expr| $e + $e |]

testTwice = twice [expr| 3 * 3|]

main = do
     print test
     print testTwice
