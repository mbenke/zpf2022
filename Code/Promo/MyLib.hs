module MyLib (someFunc) where
import qualified Promo
import qualified Singletons
import qualified TypeEquality

someFunc :: IO ()
someFunc = TypeEquality.main
