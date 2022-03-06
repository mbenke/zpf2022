module Tree91 where
import Test.QuickCheck
import Control.Monad

data Tree a = Leaf a | Branch (Tree a) (Tree a)
  deriving (Eq,Show)

instance Arbitrary a => Arbitrary (Tree a) where
    arbitrary = frequency
        [(1, liftM Leaf arbitrary)
        ,(2, liftM2 Branch arbitrary arbitrary)
        ]

atree :: Gen (Tree Int)
atree = arbitrary

threetrees :: Gen [Tree Int]
threetrees = sequence [arbitrary, arbitrary, arbitrary]
