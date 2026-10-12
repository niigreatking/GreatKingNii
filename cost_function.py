import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#For each company i, it predict the sales and subtract it from the actual sales,
#Squares the difference and sum everything. Total error is divided the number of companies
def cost_function(radio, sales, weight, bias):
    companies = len(radio)
    total_error=0.0
    for i in range(companies):
        total_error += (sales[i] - (weight*radio[i] + bias))**2
    return total_error / companies



# Example data
df=pd.read_csv("/Users/greatking_nii/Desktop/ML/Advertising.csv")

print(df.head())
print(df.columns)

x = df[["Radio"]].values
y = df["Sales"].values

# Trained weight and bias
weight = 0.7
bias = 0.02

#Plotting image
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color="#4a90e2", s=80, label="Actual data", zorder=3)
plt.plot(x, weight * x + bias, color="#e74c3c", linewidth=2.5, label=f"Prediction: y = {weight}x + {bias}")

plt.xlabel("Radio", fontsize=12)
plt.ylabel("Sales", fontsize=12)
plt.title("Cost Function", fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()



