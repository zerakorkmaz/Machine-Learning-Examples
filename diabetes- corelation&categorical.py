import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
from math import floor

df=pd.read_csv("diabetes_prediction_dataset.csv")
print(df.head())

#MISSING VALUE CONTROL
print(df.isnull().sum())

#missing value olmadığını görüyoruz.

#Categorical değerleri numeric hale getireceğiz.
#Female-Male, smoking history için bunu yapabiliriz.
categoric=df.dtypes==np.object_
print(categoric)
categorical_columns=df.columns[categoric]
print(categorical_columns)
#bu sutünları tespit ettik. şimdi de convert edelim.
for column in categorical_columns:
    LabelEncoder().fit(df[column].drop_duplicates())
    df[column]=LabelEncoder().fit_transform(df[column])
print(df.head().to_string())

#female=0,male=1 ve never=4,no info=0,current=1 gibi değerler atadı data set içerisinde.

#CORELATION
#diyabete en çok etki eden faktörleri saptayacağız.

print(df.corr()["diabetes"].sort_values(ascending=False))

plt.figure(dpi=100)
sns.heatmap(np.round(df.corr(),2),annot=True,cmap="coolwarm")
plt.show()print(df.head().to_string())


# En çok ilişkiye sahip 3 fakrötürun
# blood_glucose_level    0.419558
# HbA1c_level            0.400660
# age                    0.258008 olduğunu görebiliriz.


#diabetes değerleri sadece 0 ve 1 olduğundan lineer regresyona uygun bir data seti olmadı.





