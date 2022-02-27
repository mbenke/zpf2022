{-# LANGUAGE FlexibleInstances, FlexibleContexts, AllowAmbiguousTypes #-}
{-# LANGUAGE MultiParamTypeClasses, FunctionalDependencies #-}
class Collection c e where
      empty :: c
      insert :: e -> c -> c
      member :: e -> c -> Bool

instance Eq a => Collection [a] a where
  empty = []
  insert  = (:)
  member = elem

ins2 x y c = insert y (insert x c)

-- problem1 :: [Int]
-- problem1 = ins2 (1::Int) (2::Int) empty
-- problem2 :: [Char]
-- problem2 = ins2 'a' 'b' empty

problem3 :: (Collection c0 Char, Collection c0 Bool) => c0 -> c0
problem3 = ins2 True 'a'
-- typechecks, but should not
