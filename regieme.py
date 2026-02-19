import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn.hmm import GaussianHMM

# Download historical prices
data = yf.download("SPY", start="2015-01-01", end="2025-01-01")
# print(data.head())
# Columns = Close, High, Low, Open, Volume, 
volume = data['Volume']
prices= data['Close']


#divides the current price by the price 1 day ago, then takes hte log of that
# THis is daily returns
logReturns = np.log(prices / prices.shift(1)).dropna()
# print(logReturns)

#will calculate the probability using 20-day periods;
volatility = volume.rolling(window=20).std().dropna()
# print(volatility)

# Normalize volume (log scale)
volume_norm = np.log(volume / volume.rolling(20).mean())
# print(volume_norm)

# Make into one dataframe
features = pd.concat([logReturns, volatility, volume_norm], axis=1).dropna()
features.columns = ["log_return", "volatility", "volume"]

X = features.values

# print(X)

hmm = GaussianHMM(n_components=2, covariance_type="full", n_iter=1000, random_state=42)
hmm.fit(X)

# Predict hidden states
hidden_states = hmm.predict(X)

print(hidden_states)

# Compute log returns

for i in range(hmm.n_components):
    mean = hmm.means_[i]
    covar = hmm.covars_[i]
    print(f"Hidden State {i}")
    print("Mean =", mean)
    print("Covariance =", covar)
    print()