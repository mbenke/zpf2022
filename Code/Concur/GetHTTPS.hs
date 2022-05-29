module GetHTTPS (getURL) where
import Network.HTTP.Client
import Network.HTTP.Client.TLS
import Data.ByteString.Lazy (ByteString)

getURL :: String -> IO ByteString
getURL url = do
  manager <- newManager tlsManagerSettings
  request <- parseRequest url
  response <- httpLbs request manager
  return $ responseBody response
