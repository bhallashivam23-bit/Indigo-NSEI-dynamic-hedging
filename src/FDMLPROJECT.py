import yfinance as yf
import pandas as pd

# Firstly i'll download the data
spot = yf.download("INDIGO.NS", start="2018-01-01", end="2024-01-01")
hedge = yf.download("^NSEI", start="2018-01-01", end="2024-01-01")

###### Note - nifty index returns are used as a proxy for nifty futures returns 
# Now I need only closing prices for calculations
spot = spot[['Close']].rename(columns={'Close': 'Spot'})
hedge = hedge[['Close']].rename(columns={'Close': 'Hedge'})

# Now i'll Merge datasets like JOINS in SQL
data = pd.merge(spot, hedge, left_index=True, right_index=True)
data.columns = data.columns.get_level_values(0)  # ← ADD THIS LINE HERE

# Drop missing values
data = data.dropna()

print(data.head())


import numpy as np

# Calculate log returns(Logic:shift(1) moves your data down by one row. By dividing data['Spot'] by data['Spot'].shift(1), you are essentially dividing Today's Price by Yesterday's Price to find the ratio of change.)
data['Spot_Return'] = np.log(data['Spot'] / data['Spot'].shift(1))
data['Hedge_Return'] = np.log(data['Hedge'] / data['Hedge'].shift(1))
#Also hedge_returns is the return on hedge instrument(futures) and we havent taken any position yet

# Drop NA
data = data.dropna()

print(data.head())

import matplotlib.pyplot as plt

#Visualisation of spot vs hedge return
data[['Spot_Return', 'Hedge_Return']].plot(figsize=(10,5))
plt.title("Returns of Spot vs Hedge")
plt.show()

#correlation
print(data[['Spot_Return', 'Hedge_Return']].corr())

# Calculate covariance and variance(.iloc[0,1] 0 is for spot and 1 is for hedge hence cov matrix is between spot and hedge
cov = data['Spot_Return'].cov(data["Hedge_Return"])
var = data['Hedge_Return'].var()

# Hedge ratio (B)
beta = cov / var

print("Hedge Ratio (Beta):", beta)

data['Hedged_Return'] = data['Spot_Return'] - beta * data['Hedge_Return']

unhedged_var = data['Spot_Return'].var()
hedged_var = data['Hedged_Return'].var()
hedge_effectiveness = 1 - (hedged_var / unhedged_var)

print("Unhedged Variance:", unhedged_var)
print("Hedged Variance:", hedged_var)
print("Hedge effectiveness:", hedge_effectiveness)

######### Note - because there is only moderate corr of 39%, hedge effectivess is also low, also there is presence
######### Also there is presenece of unsystematic risk which nifty futures cannot hedged
######### we can note from the data that hedge effectiveness or r^2 is just 15% which is not very high which is quite normal in finance hence a medium effective hedge  

# Dynamic hedging 
window = 60
data['Rolling_Beta'] = data['Spot_Return'].rolling(window).cov(data['Hedge_Return']) / data['Hedge_Return'].rolling(window).var()
print(data[['Rolling_Beta']].head(70))

####### Note - above code calculates dynamic beta(60 days rolling) so for 60th day there is one beta and for 61st day there is another
####### and so on and so forth. Hence now the beta is not static it is dynamic and changes with chaning days

#Visualisation of rolling beta
data['Rolling_Beta'].plot(figsize=(10,5))
plt.title("Rolling Hedge Ratio (Beta over time)")
plt.show()

#Hedged return
data['Rolling_Hedged_Return'] = data['Spot_Return'] - data['Rolling_Beta'] * data['Hedge_Return']

#clean NaNs
data = data.dropna()

print(data[['Rolling_Beta', 'Rolling_Hedged_Return']].head(70))

#Comparing variances of dynamic hedge and unhedged returns
rolling_var = data['Rolling_Hedged_Return'].var()
unhedged_var = data['Spot_Return'].var()

rolling_effectiveness = 1 - (rolling_var / unhedged_var)

print("Rolling Hedge Effectiveness:", rolling_effectiveness)

###### Note - as we compare the hedge effectiveness in case of static andd dynamic betas, rolling hedge has 400 bps more effectivesness
###### as compared to static hedge

###### Up untill now we have calculated betas in static and dynamic(rolling) form. But now we'll train model and predict betas instead of 
###### of manual


#ML based hedge ratio
#INPUTS --
# We give model signals like:
# Rolling volatility (spot)
# Rolling volatility (hedge)
# Rolling correlation
# Lagged returns
# volatality ratio
# Basis (Spot - Futures)

# Rolling volatility (spot)
data["Spot_Vol"] = data['Spot_Return'].rolling(window).std()
data['Hedge_Vol'] = data['Hedge_Return'].rolling(window).std()

# Rolling correlation
data["Rolling_Corr"] = data['Spot_Return'].rolling(window).corr(data['Hedge_Return'])

# Lagged returns (yesterday's movement as a signal)
data["Lag_Spot_1"]  = data["Spot_Return"].shift(1)
data["Lag_Hedge_1"] = data["Hedge_Return"].shift(1)

# Vol ratio (directly encodes σs/σh — the OLS formula itself)
data["Vol_Ratio"] = data["Spot_Vol"] / data["Hedge_Vol"]

# Basis (price difference — mean reverts, signals regime)
data["Basis"] = data["Spot"] - data["Hedge"]

# Drop NaN after adding features
data = data.dropna()
import sklearn
from sklearn.model_selection import train_test_split

# Target
data["Future_Beta"] = data["Rolling_Beta"].shift(-1)

# Drop NaNs BEFORE creating X and Y
data = data.dropna()

# Now create X and Y (IMPORTANT: AFTER dropna)
X = data[["Spot_Vol", "Hedge_Vol", "Rolling_Corr",
          "Lag_Spot_1", "Lag_Hedge_1", "Vol_Ratio", "Basis"]]
Y = data["Future_Beta"]

#Train-Test split(80%)
split = int(0.8*len(data))

######## Note- normal train test split in python looks like train_test_split(X, y, shuffle=True). But as i have time series financial data this would completely be wrong
######## Because You are using future data to predict past
# Example:
# Train uses 2023 data
# Test uses 2020 data
# This is data leakage

X_train = X[:split]
X_test = X[split:]

Y_train = Y[:split]
Y_test = Y[split:]

from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(
    n_estimators=200,   # more trees
    max_depth=5,        # slightly deeper
    min_samples_leaf=10,
    random_state=42
)
model.fit(X_train, Y_train)

#Predict beta
ML_data = data.loc[X_test.index].copy()
ML_data['ML_Beta'] = model.predict(X_test)


ML_data["ML_Hedged_Return"] = ML_data['Spot_Return'] - ML_data["ML_Beta"]*ML_data['Hedge_Return']

hedged_ML_Var = ML_data['ML_Hedged_Return'].var()
Unhedged_Var = ML_data["Spot_Return"].var()

ML_Hedge_effectiveness = 1 - (hedged_ML_Var /Unhedged_Var)

print("ML Hedge Effectiveness:", ML_Hedge_effectiveness)




