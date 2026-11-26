from turtle import color
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report

import datetime
print (f'Last updated:{datetime.datetime.now()}\n')

df = pd.read_csv("https://raw.githubusercontent.com/mrdbourke/zero-to-mastery-ml/master/data/heart-disease.csv")
print(df.shape)
print(df.head(10))
print(df.target.value_counts())
print (df.target.value_counts().plot(kind="bar", color=["red", "lightblue"]))
plt.show()

print(df.info(),df.describe())
print(df.sex.value_counts())
print(pd.crosstab(index=df.target, columns=df.sex))
pd.crosstab(index=df.target, columns=df.sex).plot(kind="bar", color=["red", "lightblue"])
plt.title("Heart Disease Freq by Sex")
plt.xlabel("0 = No Disease, 1 = Disease")
plt.ylabel("Amount")
plt.legend(["Female", "Male"])
plt.xticks(rotation=0)
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(df.age[df.target==1], df.thalach[df.target==1], color="red")
plt.scatter(df.age[df.target==0], df.thalach[df.target==0], color="lightblue")
plt.xlabel("Age")
plt.ylabel("Maximum Heart Rate")
plt.title("Heart Disease by Age and Max Heart Rate")
plt.legend(["Disease", "No Disease"])
plt.show()

pd.crosstab(df.cp, df.target).plot(kind="bar", figsize=(10, 6), color=["lightblue", "red"])
plt.title("Heart Disease Frequency / Chest Pain Type")
plt.xlabel("Chest Pain Type")
plt.ylabel("Frequency")
plt.xticks(rotation=0)
plt.legend(["No Disease", "Disease"])
plt.show()

# Finding the correlation between Independent Variables
corr_matrix = df.corr()
plt.figure(figsize=(15, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="YlGnBu")
plt.title("Correlation Matrix") 
plt.show()