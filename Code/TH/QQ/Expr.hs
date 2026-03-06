{-# LANGUAGE DeriveDataTypeable #-}
module Expr where
import Text.ParserCombinators.Parsec
import Text.ParserCombinators.Parsec.Error(ParseError)
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
         | EVar String
           deriving(Show,Typeable,Data)

lexeme :: Parser a -> Parser a
lexeme = (<* spaces)

-- pNum and pIdent simplified to one char only
pNum :: Parser Expr
pNum = fmap (EInt . digitToInt) (lexeme digit)

pIdent :: Parser Expr
pIdent = EVar . (:[]) <$> (lexeme letter)

pExpr = pTerm `chainl1` spaced addop
addop :: Parser (Expr->Expr->Expr)
addop   = fmap (const EAdd) (char '+')
          <|> fmap (const ESub) (char '-')

pFactor = pNum <|> pIdent

pTerm = pFactor `chainl1` spaced mulop
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

test1, test2 :: Either ParseError Expr
test1 = parse pExpr "test1" "1 - 2 - 3 * 4 "
test2 = parse pExpr "test2" "0 + z"

parseExpr :: MonadFail m => (String, Int, Int) -> String -> m Expr
parseExpr (file, line, col) s =
    case runParser (spaces *> p) () "" s of
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
