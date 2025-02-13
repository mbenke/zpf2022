import Control.Monad.ST
import Data.STRef
fibST :: Integer -> Integer
fibST n =
    if n < 2 then n else runST (fib2 n)

fib2 :: Integer -> ST s Integer
fib2 n =  do
  x <- newSTRef 0
  y <- newSTRef 1
  fib3 n x y

fib3 0 x _ = readSTRef x
fib3 n x y = do
        x' <- readSTRef x
        y' <- readSTRef y
        writeSTRef x y'
        writeSTRef y (x'+y')
        fib3 (n-1) x y
