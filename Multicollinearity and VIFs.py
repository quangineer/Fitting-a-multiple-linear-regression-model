import pandas as pd 
import numpy as np 
import seaborn as sb 
import matplotlib.pyplot as plt
from patsy import dmatrices 
import statsmodels.api as sm 
from statsmodels.stats.outliers_influence import variance_inflation_factor

df = pd.read_csv("house_prices.csv")

# View relationship between each variable (each column) by seaborn: 
sb.pairplot(df[["area", "bedrooms", "bathrooms"]])
# plt.show()
# We can see the very positive between three variables above

df['intercept'] = 1

lm = sm.OLS(df['price'], df[['intercept', 'area', 'bedrooms', 'bathrooms']])
results = lm.fit()
# print(results.summary())
# After fitting three variable above into a multiple linear regression model, we see a negative relationship
# This is a side effect of having multicollinearity in our model.

sb.pairplot(df[['price', 'bedrooms']])
plt.show()
# Variance Inflation Factor (VIF):
y, X = dmatrices('price ~ area + bedrooms + bathrooms', df, return_type = 'dataframe')

vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns 
print(vif)