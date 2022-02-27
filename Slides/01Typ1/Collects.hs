{-# LANGUAGE FlexibleInstances, FlexibleContexts #-}
{-# LANGUAGE MultiParamTypeClasses, FunctionalDependencies #-}
class Collects e ce | ce -> e where
      empty  :: ce
      insert :: e -> ce -> ce
      member :: e -> ce -> Bool

instance Eq a => Collects a [a] where
  empty = []
  insert  = (:)
  member = elem

ins2 x y c = insert y (insert x c)

problem1 :: [Int]
problem1 = ins2 1 2 []
problem2 = ins2 'a' 'b' []

problem3 :: (Collects c0 Char, Collects c0 Bool) => c0 -> c0
problem3 = ins2 True 'a'
-- Couldn't match expected type ‘Bool’ with actual type ‘Char’
