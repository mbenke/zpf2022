{-
    Excerpted from BNF Converter BNFC.Utils module
    Copyright (C) 2004  Author:  Aarne Ranta

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
-}

module MkName where
import Control.Arrow ((&&&))
import Data.Char
import Data.List (intercalate)

-- | Different case style
data NameStyle = LowerCase  -- ^ e.g. @lowercase@
               | UpperCase  -- ^ e.g. @UPPERCASE@
               | SnakeCase  -- ^ e.g. @snake_case@
               | CamelCase  -- ^ e.g. @CamelCase@
               | MixedCase  -- ^ e.g. @mixedCase@
  deriving (Show, Eq)

-- | Generate a name in the given case style taking into account the reserved
-- word of the language. Note that despite the fact that those name are mainly
-- to be used in code rendering (type Doc), we return a String here to allow
-- further manipulation of the name (like disambiguation) which is not possible
-- in the Doc type.
-- Examples:
-- >>> mkName [] LowerCase "FooBAR"
-- "foobar"
-- >>> mkName [] UpperCase "FooBAR"
-- "FOOBAR"
-- >>> mkName [] SnakeCase "FooBAR"
-- "foo_bar"
-- >>> mkName [] CamelCase "FooBAR"
-- "FooBAR"
-- >>> mkName [] CamelCase "Foo_bar"
-- "FooBar"
-- >>> mkName [] MixedCase "FooBAR"
-- "fooBAR"
-- >>> mkName ["foobar"] LowerCase "FooBAR"
-- "foobar_"
-- >>> mkName ["foobar", "foobar_"] LowerCase "FooBAR"
-- "foobar__"
mkName :: [String] -> NameStyle -> String -> String
mkName reserved style s = notReserved name'
  where
    notReserved s
      | s `elem` reserved = notReserved (s ++ "_")
      | otherwise = s
    tokens = parseIdent s
    name' = case style of
        LowerCase -> map toLower (concat tokens)
        UpperCase -> map toUpper (concat tokens)
        CamelCase -> concatMap capitalize tokens
        MixedCase -> case concatMap capitalize tokens of
                         "" -> ""
                         c:cs -> toLower c:cs
        SnakeCase -> map toLower (intercalate "_" tokens)
    capitalize [] = []
    capitalize (c:cs) = toUpper c:cs


-- | Heuristic to "parse" an identifier into separate componennts
--
-- >>> parseIdent "abc"
-- ["abc"]
--
-- >>> parseIdent "Abc"
-- ["Abc"]
--
-- >>> parseIdent "WhySoSerious"
-- ["Why","So","Serious"]
--
-- >>> parseIdent "why_so_serious"
-- ["why","so","serious"]
--
-- >>> parseIdent "why-so-serious"
-- ["why","so","serious"]
--
-- Some corner cases
-- >>> parseIdent "LBNFParser"
-- ["LBNF","Parser"]
--
-- >>> parseIdent "ILoveNY"
-- ["I","Love","NY"]
parseIdent :: String -> [String]
parseIdent = p [] . map (classify &&& id)
  where
    classify c
        | isUpper c = U
        | isLower c = L
        | otherwise = O
    p [] [] = []
    p acc [] = reverse acc: p [] []
    p [] ((L,c):cs) = p [c] cs
    p [] ((U,c):cs) = p [c] cs
    p [] ((O,_):cs) = p [] cs
    p acc ((L,c1):cs@((L,_):_)) = p (c1:acc) cs
    p acc ((U,c1):cs@((L,_):_)) = reverse acc:p [c1] cs
    p acc ((U,c1):cs@((U,_):_)) = p (c1:acc) cs
    p acc ((L,c1):cs@((U,_):_)) = reverse (c1:acc) : p [] cs
    p acc ((U,c1):(O,_):cs) = reverse (c1:acc) : p [] cs
    p acc ((L,c1):(O,_):cs) = reverse (c1:acc) : p [] cs
    p acc ((O,_):cs) = reverse acc : p [] cs
    p acc [(_,c)] = p (c:acc) []

data CharClass = U | L | O
