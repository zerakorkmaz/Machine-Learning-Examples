import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore, skew, boxcox
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,f1_score

df = pd.read_csv("bank-additional-full.csv", sep=';')

print(df.isnull().sum())

label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

df_features=df.copy()
df_features.drop(columns=["y"],axis=1,inplace=True)
y=df["y"].values


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

#çarpık değerleri normal dağılıma yaklaştıracağım
#ama verilerim negatif değerleri içerdiği için önce hepsini pozitif yapıyorum
numeric_cols = df_features.select_dtypes(include=["int64", "float64"]).columns
for col in numeric_cols:
    if (df_features[col] <= 0).any():
        min_val = df_features[col].min()
        df_features[col] = df_features[col] + abs(min_val) + 1

negative_cols = [col for col in df_features.select_dtypes(include=["int64", "float64"]).columns if (df_features[col] < 0).any()]
print("Sıfırdan küçük değer içeren sütunlar:", negative_cols)

#çarpıklığı yüksek olan sütunlara Box-Cox uygulanır
#çarpıklıkları 0.5ten büyük olanlara bunu uygulayacağım

skewed_columns=result_df[result_df["Skewness"]>0.5].index
for col in skewed_columns:
    df_features[col], _ =boxcox(df_features[col])

result_df_after_boxcox = calculate_skewness(df_features)
print(result_df_after_boxcox)

X = df_features
y = df["y"].values

# train ve test verilerini ayırıyorum
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#aşağıdaki işlemler ile kaç komşuya bakmam durumunda en iyi sonucu alacağımı tespit ediyorum
max_k = 40
f1_scores = list()
error_rates = list()
for k in range(1, max_k):
    knn = KNeighborsClassifier(n_neighbors=k, weights='distance', metric='euclidean', p=2)
    knn = knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    f1_scores.append((k, round(f1, 4)))
    error = 1-round(accuracy_score(y_test, y_pred), 4)
    error_rates.append((k, error))

f1_result = pd.DataFrame(f1_scores, columns=['K', 'F1 Score'])
error_results = pd.DataFrame(error_rates, columns=['K', 'Error Rate'])

sns.set_context('talk')
sns.set_style('ticks')
plt.figure(dpi=300)

ax = f1_result.set_index('K').plot(color='b', figsize=(14, 7), linewidth=2)
ax.set(xlabel='K', ylabel='F1 Score')
ax.set_xticks(range(1, max_k, 2))
plt.title('KNN F1 Score')
plt.show()

#yukarıdaki kodun çıktısında oluşan grafikten göreceğim üzere 6 olduğunda en iyi sonucu alabilirim bu yüzden 6 kullanacağım
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train_scaled, y_train)
y_pred = knn.predict(X_test)

print(classification_report(y_test, y_pred))
print(f'Accuracy Score: {round(accuracy_score(y_test, y_pred), 2)}')
print(f'F1 Score: {round(f1_score(y_test, y_pred), 2)}')

cm = confusion_matrix(y_test, y_pred, labels=knn.classes_)
TN = cm[0][0]
FN = cm[1][0]
TP = cm[1][1]
FP = cm[0][1]

print(f'TN: {TN}\nFN: {FN}\nTP: {TP}\nFP: {FP}')

# jsi
jsi = TP / (TP + FP + FN)
print(f'JSI: {jsi:.2f}')
