{-# LANGUAGE QuasiQuotes #-}
module Main where
 
import Str
 
foo = [str|This is a multiline string.
It's many lines long.
 
 
It contains embedded newlines. And Unicode:
 
Pójdź, kiń-że tę chmurność w głąb flaszy.
 
It ends here: |]
 
main = putStrLn foo
