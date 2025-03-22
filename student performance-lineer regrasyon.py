import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
from math import floor

df=pd.read_csv("Student_Performance.csv")
print(df.head())

print(df.isnull().sum())

categoric=df.dtypes==np.object_
print(categoric)
categorical_columns=df.columns[categoric]
print(categorical_columns)
for column in categorical_columns:
    LabelEncoder().fit(df[column].drop_duplicates())
    df[column]=LabelEncoder().fit_transform(df[column])
print(df.head().to_string())

print(df.corr()["Performance Index"].sort_values(ascending=False))

plt.figure(dpi=100)
sns.heatmap(np.round(df.corr(),2),annot=True,cmap="coolwarm")
plt.show()

new_df=df[["Performance Index","Previous Scores"]]
print(new_df.head())


#Split Data to Train&Test
split_data=np.random.rand(len(df))<0.8
train_df=new_df[split_data]
test_df=new_df[~split_data]
print(train_df.shape,test_df.shape)


#Train
reg=linear_model.LinearRegression()
train_x=np.asarray(train_df[["Previous Scores"]])
train_y=np.asarray(train_df[["Performance Index"]])
reg.fit(train_x,train_y)
print("cofficient: %.2f" %reg.coef_[0][0])
print("intercept: %.2f" %reg.intercept_[0])


#grafik
plt.scatter(train_df["Previous Scores"],train_df["Performance Index"],color="blue")
plt.plot(train_x,reg.coef_[0][0]*train_x+reg.intercept_[0],color="red")
plt.xlabel("Performance Index")
plt.ylabel("Previous Scores")
plt.title("Performance Index vs. Previous Scores")
plt.show()

#model sonucu testi
test_x = np.asarray(test_df[["Previous Scores"]])
test_y = np.asarray(test_df[["Performance Index"]])
test_prediction = reg.predict(test_x)
print(test_prediction)
print('r2 score: %.2f' % r2_score(test_prediction, test_y))
