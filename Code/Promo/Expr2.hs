module Expr2 where

data Expr = I Int
          | B Bool
          | Add Expr Expr
          | Eq  Expr Expr

-- | eval
-- >>> eval (Add (I 2) (I 2))
-- Just (Left 4)

eval :: Expr -> Maybe (Either Int Bool)
eval (I n)       = Just (Left n)
eval (B n)       = Just (Right n)
eval _ = undefined       -- Exercise
