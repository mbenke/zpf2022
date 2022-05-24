import System.Environment

type PartialSolution = [RowNum] -- list of queen rows
type Solution = PartialSolution
type BoardSize = Int
type RowNum = Int

nqueens :: BoardSize -> [Solution]
nqueens n = subtree n [] where
  children :: PartialSolution -> [PartialSolution]
  children s = [x : s | x <- [1..n], safe x s 1]

  -- valid boards starting from given board by adding c columns
  subtree :: Int -> PartialSolution -> [PartialSolution]
  subtree 0 b = [b]
  subtree c b = concat $ map (subtree (c-1)) (children b)

safe :: Int -> PartialSolution -> BoardSize -> Bool
safe x [] n = True
safe x (c : y) n = x /= c && x /= c + n
       && x /= c - n && safe x y (n + 1)

main = do
  [s] <- getArgs
  let size = read s
  print $ length $ nqueens size
