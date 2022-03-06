module Tree1 where
import SimpleCheck1
import Control.Monad

data Tree a = Leaf a | Branch (Tree a) (Tree a)
  deriving (Eq,Show)
instance Arbitrary a => Arbitrary (Tree a) where
    arbitrary = sized arbTree

arbTree 0 = liftM Leaf arbitrary
arbTree n = frequency
        [(1, liftM Leaf arbitrary)
        ,(4, liftM2 Branch (arbTree (div n 2))(arbTree (div n 2)))
        ]

atree :: Gen (Tree Int)
atree = arbitrary

threetrees :: Gen [Tree Int]
threetrees = sequence [atree, atree, atree]