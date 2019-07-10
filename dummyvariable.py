import pandas as pd 
import numpy as np 
import statsmodels.api as sm
import matplotlib.pyplot as plt; 

df = pd.read_csv("house_prices.csv")

df[["A", "B", "C"]] = pd.get_dummies(df["neighborhood"])  # Create a dummy + add column A B C into dataframe

# Another (long) way to do:
# neighborhood_dummies = pd.get_dummies(df["neighborhood"])
# df_new = df.join(neighborhood_dummies)
# print (df_new.head(5))

df['intercept'] = 1
#ALL:
lm = sm.OLS(df["price"], df[["intercept", "A", "B", "C"]])
results = lm.fit()

#drop 1 "A" :
lm = sm.OLS(df["price"], df[['intercept', "B", "C"]])
results = lm.fit()
print (results.summary())

plt.hist(df.query("C == 1")['price'], alpha = 0.3, label = 'C');
plt.hist(df.query("A == 1")['price'], alpha = 0.3, label = 'A');
plt.hist(df.query("B == 1")['price'], alpha = 0.3, label = 'B');
plt.legend(loc='best')
plt.show();


# df[["lodge", "ranch", "victorian"]] = pd.get_dummies(df["style"])
# # print (df.head(5))

# df['intercept'] = 1 
# lm = sm.OLS(df["price"], df[['intercept','lodge','ranch']])
# results = lm.fit()
# print (results.summary()) 