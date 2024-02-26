import Sudoku
import Control.Exception
import System.Environment
import Control.Parallel
import Control.Seq as Seq
import Control.DeepSeq
import Data.Maybe

main :: IO ()
main = do
    [f] <- getArgs
    grids <- fmap lines $ readFile f
    -- runEval (parMap solve grids) `deepseq` return ()
    -- let solutions = runEval (parMap solve grids)
    print  . length  . filter isJust $ parMap solve grids

parMap :: (a -> b) -> [a] -> [b]
parMap f [] = []
parMap f (a:as) = par b bs `pseq` (b:bs) where
    b = f a
    bs = parMap f as
