{-# LANGUAGE DeriveDataTypeable #-}
module Expr where
import Text.ParserCombinators.Parsec
import Data.Char(digitToInt)
import Data.Typeable
import Data.Data
import Control.Applicative((<$>),(*>),(<*))
import Control.Monad.Fail(MonadFail)

data Expr = EInt Int
         | EAdd Expr Expr
         | ESub Expr Expr
         | EMul Expr Expr
         | EDiv Expr Expr
           deriving(Show,Typeable,Data)

pNum :: Parser Expr
pNum = fmap (EInt . digitToInt) digit


pExpr = pTerm `chainl1` spaced addop
addop :: Parser (Expr->Expr->Expr)
addop   =   fmap (const EAdd) (char '+')
          <|> fmap (const ESub) (char '-')

pTerm = spaced pNum `chainl1` spaced mulop
mulop :: Parser (Expr->Expr->Expr)
mulop = pOps [EMul,EDiv] ['*','/']

pOps :: [a] -> [Char] -> Parser a
pOps fs cs = foldr1 (<|>) $ map pOp $ zip fs cs

whenP :: a -> Parser b -> Parser a
whenP = fmap . const

spaced :: Parser a -> Parser a
spaced p = spaces *> p <* spaces

pOp :: (a,Char) -> Parser a
pOp (f,s) = f `whenP` char s

test1 = parse pExpr "test1" "1 - 2 - 3 * 4 "

parseExpr :: MonadFail m => (String, Int, Int) -> String -> m Expr
parseExpr (file, line, col) s =
    case runParser p () "" s of
      Left err  -> fail $ show err
      Right e   -> return e
  where
    p = do updatePosition file line col
           spaces
           e <- pExpr
           spaces
           eof
           return e

updatePosition file line col = do
   pos <- getPosition
   setPosition $
     (flip setSourceName) file $
     (flip setSourceLine) line $
     (flip setSourceColumn) col $
     pos
