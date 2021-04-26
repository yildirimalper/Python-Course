import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

dataframe = pd.read_csv("../Homework 2/RegressionData_HW2.csv")

dataframe.head()

fig = sns.lmplot(x="Average per capita transfer - All Social Assistance", y="Life expectancy at birth, total (years)", data=dataframe)
fig.set_axis_labels("Average per capita transfer of social assistance", "Life expectancy at Birth")
sns.set_style("whitegrid")
plt.title("Regression Results")
plt.savefig("Regression Results.png")
plt.show()