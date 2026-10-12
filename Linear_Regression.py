
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df=pd.read_csv("/Users/greatking_nii/Desktop/ML/Advertising.csv")

print(df.head())
print(df.columns)

x = df[["Radio"]].values
y = df["Sales"].values

model = LinearRegression()
model.fit(x, y)

plt.figure(figsize=(8,6))
plt.scatter(df["Radio"], df["Sales"], color="blue", alpha=0.7, s=40)

x_range = np.linspace(df["Radio"].min() -5, df["Radio"].max() + 5, 100)
plt.plot(x_range, model.predict(x_range.reshape(-1, 1)), color="blue", linewidth=2)

plt.title("Line Regression Line")
plt.xlabel("Radio")
plt.ylabel("Label")
plt.xlim(-10, 60)
plt.ylim(-5,30)
plt.tight_layout()
plt.show()

def predict_sales(radio, weight, bias):
    return weight*radio +bias


