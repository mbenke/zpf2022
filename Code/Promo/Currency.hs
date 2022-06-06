{-# LANGUAGE GeneralizedNewtypeDeriving #-}

data USD
data EUR

newtype Amount a = Amount Double
                 deriving (Show, Eq, Ord, Num)

usd :: Double -> Amount USD
usd = Amount

eur :: Double -> Amount EUR
eur = Amount
