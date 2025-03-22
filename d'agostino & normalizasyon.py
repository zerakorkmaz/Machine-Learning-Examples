import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import normaltest
from sklearn.datasets import load_iris
from sklearn.preprocessing import PowerTransformer, FunctionTransformer

# Veri seti yükleyeceğim
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)

# Sadece bir özellik seçelim (örneğin, 'sepal length (cm)')
X = df['sepal length (cm)']

# Normalizasyon işlemleri
pt = PowerTransformer(method='box-cox', standardize=False)
X_boxcox = pt.fit_transform(X.values.reshape(-1, 1)).flatten()

X_log = np.log(X)

X_sqrt = np.sqrt(X)

# D'Agostino K^2 testi uygulama
def dagostino_test(x, name):
    stat, p = normaltest(x)
    print(f"{name} D'Agostino K^2 testi: p-değeri = {p}")
    if p < 0.5:
        print(f"{name} başarılı (p < 0.5)")
    else:
        print(f"{name} başarısız (p >= 0.5)")


dagostino_test(X, 'Orijinal Veri')

dagostino_test(X_boxcox, 'Box-Cox')

dagostino_test(X_log, 'Log')

dagostino_test(X_sqrt, 'Square Root')

results = {
    'Orijinal Veri': normaltest(X)[1],
    'Box-Cox': normaltest(X_boxcox)[1],
    'Log': normaltest(X_log)[1],
    'Square Root': normaltest(X_sqrt)[1]
}

results_df = pd.DataFrame(list(results.items()), columns=['Dönüşüm Türü', 'p-değeri'])
print(results_df)