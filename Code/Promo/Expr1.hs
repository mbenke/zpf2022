module Expr1 where
data Expr = I Int
          | Add Expr Expr

-- | eval
-- >>> eval (Add (I 2) (I 2))
-- 4

eval :: Expr -> Int
eval (I n)       = n
eval (Add e1 e2) = eval e1 + eval e2
