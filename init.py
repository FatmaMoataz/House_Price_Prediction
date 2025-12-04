import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

df = pd.read_csv('dataset/data.csv')
# Take a first look
print(df.head())
print(df.shape)
print(df.columns)
print(df.info())

submission_example = pd.read_csv("dataset/output.csv")
print(submission_example.head())
print(submission_example.columns)

y=df['price']
numeric_features=['bedrooms' , 'bathrooms' , 'floors' , 'waterfront' , 'view']
x=df[numeric_features]
print(y.shape)
print(x.shape)

X_train , X_valid , y_train , y_valid = train_test_split(x,y,test_size=0.2,random_state=42)
print(X_train.shape)
print(X_valid.shape)
print(y_train.shape)
print(y_valid.shape)

model=LinearRegression()
model.fit(X_train,y_train)
y_predict=model.predict(X_valid)

mse = mean_squared_error(y_valid, y_predict)
rmse = np.sqrt(mse)

print("MSE:", mse)
print("RMSE:", rmse)
# Scatter plot
plt.figure(figsize=(8,6))
sns.scatterplot(data=df, x='sqft_living', y='price')
plt.show()
# Heatmap
numeric_df = df[numeric_features + ['price']]
corr = numeric_df.corr()
plt.figure(figsize=(12,8))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.show()
# Boxplot
plt.figure(figsize=(8,6))
sns.boxplot(data=df, x='view', y='price')
plt.show()
# Historical plot
plt.figure(figsize=(10,6))
sns.histplot(df['price'], kde=True)
plt.show()
