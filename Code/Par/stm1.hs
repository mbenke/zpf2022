import Control.Concurrent
import Control.Concurrent.STM

incRef :: TVar Int -> IO ()
incRef var = atomically $ do
                val <- readTVar var
                let x = fromInteger $ delay baseDelay
       	        writeTVar var (val+1+x)

main = do
  px <- newTVarIO 0     -- create top-level TVar
  mapM forkIO $ replicate 20 (incRef px)
  delay (30*baseDelay) `seq` return ()
  atomically (readTVar px) >>= print
