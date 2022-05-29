import Control.Monad.Par
import System.Environment

-- <<fib
fib :: Integer -> Integer
fib 0 = 1
fib 1 = 1
fib n = fib (n-1) + fib (n-2)
-- >>

main = do
  [n,m] <- map read <$> getArgs
  print $
-- <<runPar
    runPar $ do
      i <- new                          -- <1>
      j <- new                          -- <1>
      fork (put i (fib n))              -- <2>
      fork (put j (fib m))              -- <2>
      a <- get i                        -- <3>
      b <- get j                        -- <3>
      return (a+b)                      -- <4>
-- >>
