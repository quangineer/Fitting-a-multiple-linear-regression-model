import pandas as pd 
import numpy as np 
import statsmodels.api as sm 

df = pd.read_csv("house_prices.csv")
print (df.head())

df['intercept'] = 1
lm = sm.OLS(df['price'], df[['intercept', 'bathrooms', 'bedrooms', 'area']])
results = lm.fit()
print (results.summary())