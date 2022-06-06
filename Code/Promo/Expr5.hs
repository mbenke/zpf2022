{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE GADTs #-}
module Expr5 where

data Expr a where
  I :: Int -> Expr Int
  B :: Bool -> Expr Bool
  Add :: Expr Int -> Expr Int -> Expr Int
  Eq  :: Eq a => Expr a -> Expr a -> Expr Bool
  -- exercise: allow comparing booleans, e.g `Eq (B True) (B True)`

-- | eval
-- >>> eval $ Eq (Add (I 2) (I 2)) (I 4)
-- True
-- >>> eval $ Eq (B True) (B True)
-- True

eval :: Expr a -> a
eval (I n)       = n
eval (B b)       = b
eval (Add e1 e2) = eval e1 + eval e2
eval (Eq  e1 e2) = eval e1 == eval e2

deriving instance Show (Expr a)
