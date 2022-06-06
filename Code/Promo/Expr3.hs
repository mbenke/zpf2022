module Expr3 where

data Expr a = I Int
            | B Bool
            | Add (Expr Int) (Expr Int)
            | Eq  (Expr Int) (Expr Int)


e1 = Add (B True) (I 1)

-- e2 = Add
eval :: Expr a -> a
-- eval (I n)       = n -- error
eval _ = undefined
