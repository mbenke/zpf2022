-- (c) Simon Marlow 2011, see the file LICENSE for copying terms.
--
-- Sample geturls.hs (CEFP summer school notes, 2011)
--
-- Downloading multiple URLs concurrently, timing the downloads
--
-- Compile with:
--    ghc -threaded --make geturls.hs

import GetHTTPS(getURL)
import TimeIt

import Control.Monad
import Control.Concurrent
import Control.Exception
import Text.Printf
import qualified Data.ByteString.Lazy as BL

-----------------------------------------------------------------------------
-- Our Async API:

data Async a = Async (MVar a)

async :: IO a -> IO (Async a)
async action = do
   var <- newEmptyMVar
   forkIO (action >>= putMVar var)
   return (Async var)

wait :: Async a -> IO a
wait (Async var) = readMVar var

-----------------------------------------------------------------------------

sites = ["https://www.google.com",
         "https://www.bing.com",
         "https://www.mimuw.edu.pl",
         "https://hackage.haskell.org/package/parallel",
         "https://hackage.haskell.org/package/base"
        ]

main = mapM (async.http) sites >>= mapM wait
 where
   http url = do
     (page, time) <- timeit $ getURL url
     printf "downloaded: %s (%d bytes, %.2fs)\n" url (BL.length page) time
