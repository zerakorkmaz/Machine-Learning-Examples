import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore, skew, boxcox
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, f1_score, ConfusionMatrixDisplay

df=pd.read_csv("Cancer_Data.csv")
#print(df.head())

#gereksiz sütunları drop edelim.

df.drop(columns=["id","Unnamed: 32"],inplace=True)
#print(df.head())

#kanserin iyi ya da kötü huylu olduğunu gösteren sutün yanı diagnnosis'teki
#kategorik verileri sayısala çevirelim.

df["diagnosis"]=LabelEncoder().fit_transform(df["diagnosis"])
print(df.head())

#Eksik veri kontrolü
print(df.isnull().sum())
#Eksik veri çıkmadı

df_features=df.copy()
df_features.drop(columns=["diagnosis"],axis=1,inplace=True)
y=df["diagnosis"].values

#aykırı değerleri saptayalım
df_zscore=df_features.copy()
for col in df_features.select_dtypes(include=np.number).columns:
    df_zscore[col]=zscore(df_features[col])
for col in df_zscore.columns:
    outliers=df_zscore[df_zscore[col].abs()>3]
    print(f'{col} outliers number:{len(outliers)}\n')


def calculate_skewness(df: pd.DataFrame):
    results=[]

    for col in df.columns:
        skewness=skew(df[col])
        if skewness>1:
            result="değerler çarpık"
        elif 0.5<skewness<1:
            result="değerler orta derecede çarpık"
        elif skewness<0.5:
            result="değerler normal dağılıma yakın"

        results.append({"Features": col, "Skewness": skewness, "Result": result})

    skewed_df=pd.DataFrame(results)
    skewed_df.set_index("Features", inplace=True)
    return skewed_df


result_df=calculate_skewness(df_features)
print(result_df)

print("Sıfır Değerleri:\n", df_features[df_features == 0].count())
df_features = df_features.replace(0, 1e-6)
df_features[df_features == 0].count()
print("Güncel Sıfır Değerleri:\n", df_features[df_features == 0].count())

df_log=np.log(df_features)
result_df=calculate_skewness(df_log)
print(result_df)

X=df_log[['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
       'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
       'fractal_dimension_se', 'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave points_worst',
       'symmetry_worst', 'fractal_dimension_worst']].values

X=StandardScaler().fit_transform(X.astype(float))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
print(f'Features Train Set: {X_train.shape}\n')
print(f'Features Test Set: {X_test.shape}\n')
print(f'Target Train Set: {y_train.shape}\n')
print(f'Target Test Set: {y_test.shape}\n')


knn = KNeighborsClassifier(n_neighbors=3)
knn=knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)

print(classification_report(y_test, y_pred))
print(f'Accuracy Score: {round(accuracy_score(y_test, y_pred), 2)}')
print(f'F1 Score: {round(f1_score(y_test, y_pred), 2)}')
