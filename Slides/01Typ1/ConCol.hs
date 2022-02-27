module ConCol where

class Collection c where
  empty  :: c e
  insert :: e -> c e -> c e
  member :: Eq e => e -> c e-> Bool

instance Collection [] where
    empty = []
    insert x xs = x:xs
    member = elem

ins2 :: Collection c => e -> e -> c e -> c e           
ins2 x y c = insert y (insert x c)


noproblem1 :: [Int]
noproblem1 = ins2 1 2 []

noproblem2 = ins2 'a' 'b' []
