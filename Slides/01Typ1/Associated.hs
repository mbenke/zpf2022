{-# LANGUAGE TypeFamilies #-}
class Collection c where
      type Elem c :: *
      empty :: c
      insert :: Elem c -> c -> c
      member :: Elem c -> c -> Bool

instance Eq a => Collection [a] where
  type Elem [a] = a
  empty = []
  insert  = (:)
  member = elem

ins2 :: Collection c => Elem c -> Elem c -> c -> c
ins2 x y c = insert y (insert x c)

noproblem1 :: [Int]
noproblem1 = ins2 (1::Int) (2::Int) empty

noproblem2 :: [Char]
noproblem2 = ins2 'a' 'b' empty

-- does not typecheck
-- noproblem3 :: (Collection c0 Char, Collection c0 Bool) => c0 -> c0
-- noproblem3 = ins2 True 'a'

