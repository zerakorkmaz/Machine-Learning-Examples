import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
from math import floor
from mpl_toolkits.mplot3d import Axes3D

df=pd.read_csv("housing.csv")
print(df.head())

#MISSING VALUE CONTROL
print(df.isnull().sum())
#total bedroomsta missing value var. onları dolduralım
df["total_bedrooms"] = df["total_bedrooms"].fillna(df["total_bedrooms"].mean()).astype(int)
print(df.isnull().sum())
#ortalama ile missing valueları doldurdum.

#Categorical değerleri numeric yapacağız. bu data setinde "ocean_proximity" sütununu yapabiliriz.
categoric=df.dtypes==np.object_
print(categoric)
categorical_columns=df.columns[categoric]
print(categorical_columns)
for column in categorical_columns:
    LabelEncoder().fit(df[column].drop_duplicates())
    df[column]=LabelEncoder().fit_transform(df[column])
print(df.head().to_string())

#CORELATİON
corelation =df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corelation, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Corelation Matrix - Housing Dataset")
plt.show()

#datadaki kullanmayacağım sütunları atacağım.
new_df=df[["total_bedrooms","households"]]
print(new_df.head())


#Split Data to Train&Test
split_data=np.random.rand(len(df))<0.8
train_df=new_df[split_data]
test_df=new_df[~split_data]
print(train_df.shape,test_df.shape)


#Train
reg=linear_model.LinearRegression()
train_x=np.asarray(train_df[["total_bedrooms"]])
train_y=np.asarray(train_df[["households"]])
reg.fit(train_x,train_y)
print("cofficient: %.2f" %reg.coef_[0][0])
print("intercept: %.2f" %reg.intercept_[0])

#manuel test
room=float(input("Enter total bedrooms: "))
y=reg.intercept_[0]+reg.coef_[0][0]*room
print(f"households: {floor(y)}")

#grafik
plt.scatter(train_df["households"],train_df["total_bedrooms"],color="blue")
plt.plot(train_x,reg.coef_[0][0]*train_x+reg.intercept_[0],color="red")
plt.xlabel("Households")
plt.ylabel("Total bedrooms")
plt.title("Households vs. Total bedRooms")
plt.show()

#model sonucu testi
test_x = np.asarray(test_df[["total_bedrooms"]])
test_y = np.asarray(test_df[["households"]])
test_prediction = reg.predict(test_x)
print(test_prediction)
print('r2 score: %.2f' % r2_score(test_prediction, test_y))

