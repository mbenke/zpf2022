import GetURL
import Control.Concurrent

main = do
  m1 <- newEmptyMVar
  m2 <- newEmptyMVar
  forkIO $ do
    putStrLn "START 1"
    r <- getURL "http://hackage.haskell.org/package/base"
    putMVar m1 r

  forkIO $ do
    putStrLn "START 2"
    r <- getURL "http://hackage.haskell.org/package/parallel"
    putMVar m2 r

  r1 <- takeMVar m1
  putStrLn "1 DONE"
  r2 <- takeMVar m2
  putStrLn "2 DONE"
