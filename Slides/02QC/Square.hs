{-|
Module: Square

This is a simple module to demonstrate Haddock doc comments.
-}
module Square(square) where

-- | The 'square' function squares an integer.
-- It takes one argument, of type 'Int'.
square :: Int -> Int
square x = x * x
