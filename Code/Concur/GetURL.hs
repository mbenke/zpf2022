module GetURL where
import Data.ByteString(ByteString)
import Network.HTTP
import Network.URI(parseURI)

get urlString =
  case parseURI urlString of
    Nothing -> error ("getRequest: Not a valid URL - " ++ urlString)
    Just u  -> mkRequest GET u

getURL :: String -> IO ByteString
getURL url = simpleHTTP (get url) >>= getResponseBody
