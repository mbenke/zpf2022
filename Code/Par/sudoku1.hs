import Sudoku
import Control.DeepSeq
import System.Environment
import Data.Maybe

main :: IO ()
main = do
    [f] <- getArgs
    grids <- fmap lines $ readFile f
    print $ length $ filter isJust $ map solve grids
