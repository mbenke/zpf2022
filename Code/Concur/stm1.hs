import Control.Concurrent
import Control.Concurrent.STM

incRefSTM :: TVar Int -> STM ()
incRefSTM var = do
                val <- readTVar var
                let x = fromInteger $ delay baseDelay
                writeTVar var (val+1+x)

incRef :: TVar Int -> IO ()
incRef = atomically . incRefSTM

main = do
  px <- newTVarIO 0                       -- create top-level TVar
  mapM forkIO $ replicate 20 (incRef px)  -- start 20 incRef threads
  delay (30*baseDelay) `seq` return ()
  atomically (readTVar px) >>= print

baseDelay :: Integer
baseDelay = 10^7

delay :: Integer -> Integer
delay 0 = 0
delay n = delay $! n-1
