-- #!/usr/bin/env cabal
{- cabal:
build-depends:
  base ^>=4.16.0.0,
  random ^>=1.2,
-}
module SimpleCheck1 where
import Control.Monad(ap)
import System.Random
  ( StdGen       -- :: *
  , newStdGen    -- :: IO StdGen
  , Random(..)   -- class
  , randomR      -- :: (RandomGen g, Random a) => (a, a) -> g -> (a, g)
  , split        -- :: RandomGen g => g -> (g, g)
                 -- rozdziela argument na dwa niezależne generatory
    
  -- instance RandomGen StdGen
  )
import Data.List( group, sort, intersperse )
import Control.Monad( liftM2, liftM3, liftM4 )

-- infixr 0 ==>
-- infix  1 `classify`

newtype Gen a
  = Gen (Int -> StdGen -> a)

sized :: (Int -> Gen a) -> Gen a
sized fgen = Gen (\n r -> let Gen m = fgen n in m n r)

resize :: Int -> Gen a -> Gen a
resize n (Gen m) = Gen (\_ r -> m n r)

instance Monad Gen where
  return = pure
  Gen m >>= k = Gen $ \n r0 ->
    let (r1,r2) = split r0
        Gen m'  = k (m n r1)
     in m' n r2
        
instance Functor Gen where
  fmap f m = m >>= return . f        

instance Applicative Gen where
  pure a = Gen $ \n r -> a
  (<*>) = ap

rand :: Gen StdGen
rand = Gen (\n r -> r)

chooseInt1 :: (Int,Int) -> Gen Int
chooseInt1 bounds = Gen $ \n r  -> fst (randomR bounds r)
                     
chooseInt :: (Int,Int) -> Gen Int
chooseInt bounds = (fst . randomR bounds) `fmap` rand

choose ::  Random a => (a, a) -> Gen a
choose bounds = (fst . randomR bounds) <$> rand

elements :: [a] -> Gen a
elements xs = (xs !!) `fmap` choose (0, length xs - 1)

vector :: Arbitrary a => Int -> Gen [a]
vector n = sequence [ arbitrary | i <- [1..n] ]
-- sequence :: Monad m => [m a] -> m [a]

genOne :: Gen a -> IO a
genOne (Gen m) =
  do 
    rnd0 <- newStdGen
    return $ m 7 rnd0

oneof :: [Gen a] -> Gen a
oneof gens = elements gens >>= id

frequency :: [(Int, Gen a)] -> Gen a
frequency xs = choose (1, tot) >>= (`pick` xs)
 where
  tot = sum (map fst xs)

  pick n ((k,x):xs)
    | n <= k    = x
    | otherwise = pick (n-k) xs

-- * Arbitrary
class Arbitrary a where
  arbitrary   :: Gen a

instance Arbitrary () where
  arbitrary = return ()
  
instance Arbitrary Bool where
  arbitrary     = elements [True, False]
  
instance Arbitrary a => Arbitrary [a] where
  arbitrary          = sized (\n -> choose (0,n) >>= vector)

instance Arbitrary Int where
  arbitrary     = sized $ \n -> choose (-n,n)

genInt :: IO Int
genInt = genOne arbitrary

genInts :: IO [Int]
genInts = genOne arbitrary

generate :: Int -> StdGen -> Gen a -> a
generate n rnd (Gen m) = m size rnd'
 where
  (size, rnd') = randomR (0, n) rnd

genIO :: Int -> Gen a -> IO a
genIO n g = do
   rnd <- newStdGen
   return $ generate n rnd g
   
data Result = Result { ok :: Maybe Bool, arguments :: [String] }

nothing :: Result
nothing = Result{ ok = Nothing,  arguments = [] }

newtype Property
  = Prop (Gen Result)
    
class Testable a where
  property :: a -> Property  
  
result :: Result -> Property
result res = Prop (return res)

instance Testable () where
  property () = result nothing

instance Testable Bool where
  property b = result (nothing { ok = Just b })

instance Testable Property where
  property prop = prop

evaluate :: Testable a => a -> Gen Result
evaluate a = gen where Prop gen = property a 
                       
forAll :: (Show a, Testable b) => Gen a -> (a -> b) -> Property
forAll gen body = Prop $
  do a   <- gen
     res <- evaluate (body a)
     return (argument a res)
 where
  argument a res = res{ arguments = show a : arguments res }
  

instance (Arbitrary a, Show a, Testable b) => Testable (a -> b) where
  property f = forAll arbitrary f

(==>) :: Testable a => Bool -> a -> Property
True  ==> a = property a
False ==> a = property ()

-- Driver
check :: Testable prop => prop -> IO ()
check prop = do
  rnd <- newStdGen
  tests (evaluate prop) rnd 0 0
  
tests :: Gen Result -> StdGen -> Int -> Int -> IO () 
tests gen rnd0 ntest nfail 
  | ntest == configMaxTest = done "OK, passed" ntest
  | nfail == configMaxFail = done "Arguments exhausted after" ntest
  | otherwise               =
      do putStr (configEvery  ntest (arguments result))
         case ok result of
           Nothing    ->
             tests gen rnd1 ntest (nfail+1) 
           Just True  ->
             tests gen rnd1 (ntest+1) nfail 
           Just False ->
             putStr ( "Falsifiable, after "
                   ++ show ntest
                   ++ " tests:\n"
                   ++ unlines (arguments result)
                    )
     where
      result      = generate (configSize ntest) rnd2 gen
      (rnd1,rnd2) = split rnd0


done :: String -> Int  -> IO ()
done mesg ntest  =
  do putStrLn ( mesg ++ " " ++ show ntest ++ " tests" )

configMaxTest, configMaxFail :: Int
configMaxTest = 100
configMaxFail = 500
configSize   :: Int -> Int
configSize    = (+ 3) . (`div` 2)

configEvery  :: Int -> [String] -> String 
configEvery   = \n args -> let s = show n in s ++ [ '\b' | _ <- s ]


propAddCom1 :: Property
propAddCom1 =  forAll (chooseInt (-100,100)) (\x -> x + 1 == 1 + x)

propAddCom2 =  forAll int (\x -> forAll int (\y -> x + y == y + x)) where
  int = chooseInt (-100,100)
  
propAddCom3 :: Int -> Int -> Bool  
propAddCom3 x y = x + y == y + x

propMul1 :: Int -> Property
propMul1 x = (x>0) ==> (2*x > 0) 

propMul2 :: Int -> Int -> Property
propMul2 x y = (x>0) ==> (x*y > 0) 
