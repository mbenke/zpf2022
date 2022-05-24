import Sudoku
import Control.Exception
import System.Environment
import Data.Maybe
import Control.Parallel.Strategies
import Control.DeepSeq

main :: IO ()
main = do
    [f] <- getArgs
    grids <- fmap lines $ readFile f
    -- print (length (filter isJust (map solve grids)))
    let (as,bs) = splitAt (length grids `div` 2) grids
    print (length (runEval (work as bs)))

work as bs =  do
       a <- rpar (force (map solve as))
       b <- rpar (force (map solve bs))
       return (filter isJust (a++b))


