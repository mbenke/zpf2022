module Tree1 where
import SimpleCheck1
import Control.Monad

data Tree a = Leaf a | Branch (Tree a) (Tree a)
  deriving (Eq,Show)

instance Arbitrary a => Arbitrary (Tree a) where
    arbitrary = frequency
        [(1, Leaf <$> arbitrary)
        ,(2, Branch <$> arbitrary <*> arbitrary)
        ]

atree :: Gen (Tree Int)
atree = arbitrary

threetrees :: Gen [Tree Int]
threetrees = sequence [arbitrary, arbitrary, arbitrary]
