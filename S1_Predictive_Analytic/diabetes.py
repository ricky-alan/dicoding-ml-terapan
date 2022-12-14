# -*- coding: utf-8 -*-
"""diabetes.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1CxSf30ZJ3LqfqhQeHEuRc7jeKCM27lnn

### Data Loading
Import Library
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile

"""Load Dataset"""

!pip install -q kaggle

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 /root/.kaggle/kaggle.json

!kaggle datasets download -d alexteboul/diabetes-health-indicators-dataset

local_zip = '/content/diabetes-health-indicators-dataset.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/content')
zip_ref.close()

diabetes = pd.read_csv('/content/diabetes_binary_health_indicators_BRFSS2015.csv')
diabetes.head()

diabetes.shape

"""Dataset ini memiliki 253.680 baris dan 22 kolom.

### Exploratory Data Analysis
Deskripsi Variabel



1.   **Diabetes_binary** merupakan fitur target. Menunjukkan apakah seseorang mengidap diabetes. 0: tidak diabetes, 1: diabetes.
2.   **HighBP**: menunjukkan apakah seseorang punya darah tinggi (high blood pressure). 0: tidak, 1: ya.
3.   **HighChol**: menunjukkan apakah seseorang punya kolestrol tinggi. 0: tidak, 1: ya.
4.   **CholCheck**: menunjukkan apakah seseorang melakukan cek kolestrol dalam 5 tahun. 0: tidak, 1: ya.
5.   **BMI**: merupakan kategori berat badan seseorang.
6.   **Smoker**: menunjukkan apakah seseorang merokok setidaknya 100 batang rokok selama hidup. 0: tidak, 1: ya.
7.   **Stroke**: menunjukkan apakah seseorang mengidap stroke. 0: tidak, 1: ya.
8.   **HeartDiseaseorAttack**: menunjukkan apakah seseorang mengidap penyakit jantung. 0: tidak, 1: ya.
9.   **PhysActivity**: menunjukkan apakah seseorang melakukan aktivitas fisik (tidak termasuk pekerjaan) dalam 30 hari terakhir. 0: tidak, 1: ya.
10.  **Fruits**: menunjukkan apakah seseorang mengonsumsi buah minimal satu kali sehari. 0: tidak, 1: ya.
11.  **Veggies**: menunjukkan apakah seseorang mengonsumsi sayur minimal satu kali sehari. 0: tidak, 1: ya.
12.  **HvyAlcoholConsump**: merupakan konsumsi alkohol dalam seminggu (pria dewasa >= 14 minuman, wanita dewasa >= 7 minuman).
13.  **AnyHealthcare**: menunjukkan apakah seseorang memiliki jaminan kesehatan. 0: tidak, 1: ya.
14.  **NoDocbcCost**: menunjukkan apakah dalam 1 tahun terakhir seseorang memiliki kendala biaya saat ingin berobat ke dokter. 0: tidak, 1: ya.
15.  **GenHlth**: merupakan kondisi kesehatan secara umum. 1: sangat sangat bagus, 2: sangat bagus, 3: bagus, 4: lumayan, 5: buruk.
16.  **MentHlth**: menunjukkan berapa hari seseorang dalam kondisi mental yang buruk (0-30 hari).
17.  **PhysHlth**: merupakan luka fisik atau cidera dalam 30 hari terakhir (0-30 hari).
18.  **DiffWalk**: menunjukkan apakah seseorang punya kesulitan berjalan atau naik tangga. 0: tidak, 1: ya.
19.  **Sex**: jenis kelamin. 0: perempuan, 1: laki-laki.
20.  **Age**: kategori umur seseorang. 
21.  **Education**: tingkat pendidikan.
22.  **Income**: pendapatan.

Cek Info Data
"""

diabetes.info()

"""Semua variabel (kolom) memiliki tipe data float64

Cek Informasi Statistik
"""

diabetes.describe()

"""Cek Missing Data"""

diabetes.isnull().any().sum()

"""Tidak ada missing data pada dataset ini.

Cek Data Duplikat
"""

duplicates = diabetes[diabetes.duplicated()]
len(duplicates)

"""Terdapat 24206 baris duplikat. Baris-baris ini akan dihapus."""

diabetes.drop_duplicates(inplace=True)
diabetes.shape

"""Visualisasi"""

count = diabetes['Diabetes_binary'].value_counts()
count.plot(kind='pie', title='Diabetes_binary', labels=["non-Diabetic","Diabetic"], autopct='%.02f')

"""Sebanyak 84.71% responden tidak mengidap diabetes sedangkan 15.29% sisanya mengidap diabetes.

Univariate Analysis
"""

diabetes.hist(figsize=(16,12))

"""Dari grafik diatas, dapat diketahui distribusi data setiap variabel.

Multivariate Analysis
"""

plt.figure(figsize=(20, 15))
sns.heatmap(data=diabetes.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")

"""Dari matriks diatas, dapat diketahui bahwa fitur yang mempunyai korelasi yang kuat dengan fitur target, diantaranya HighBP, HighCol, BMI, HeartDiseaseorAttack, GenHlth, PhyscHlth, DiffWalk, dan Age.

### Data Preparation
Tran Test Split
"""

from sklearn.model_selection import train_test_split

X = diabetes.drop('Diabetes_binary', axis=1)
Y = diabetes['Diabetes_binary']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=123)

print(f'Jumlah train dataset: {len(X_train)}')
print(f'Jumlah test dataset: {len(X_test)}')

"""### Model Development

Algoritma K-Nearest Neighbors
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error, accuracy_score

knn = KNeighborsClassifier(n_neighbors=12)
knn.fit(X_train, Y_train)
knn_pred = knn.predict(X_train)

print('Accuracy Score: {}'.format(accuracy_score(y_pred = knn_pred, y_true=Y_train)))
print('Mean Squared Error: {}'.format(mean_squared_error(y_pred = knn_pred, y_true=Y_train)))

"""Algoritma Random Forest"""

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=55, n_jobs=-1)
rf.fit(X_train, Y_train)
rf_pred = rf.predict(X_train)

print('Accuracy Score: {}'.format(accuracy_score(y_pred = rf_pred, y_true=Y_train)))
print('Mean Squared Error: {}'.format(mean_squared_error(y_pred = rf_pred, y_true=Y_train)))

"""### Evaluasi Model

Algoritma K-Nearest Neighbors
"""

knn_eval = knn.predict(X_test)

print('Accuracy Score: {}'.format(accuracy_score(y_pred = knn_eval, y_true=Y_test)))
print('Mean Squared Error: {}'.format(mean_squared_error(y_pred = knn_eval, y_true=Y_test)))

"""Algoritma Random Forest"""

rf_eval = rf.predict(X_test)

print('Accuracy Score: {}'.format(accuracy_score(y_pred = rf_eval, y_true=Y_test)))
print('Mean Squared Error: {}'.format(mean_squared_error(y_pred = rf_eval, y_true=Y_test)))

"""Prediksi"""

prediksi = X_test.iloc[:5].copy()

print(knn.predict(prediksi))
print(rf.predict(prediksi))
print('y_true:\n{}'.format(Y_test[:5]))

"""Berdasarkan hasil prediksi diatas, kedua algoritma dapat memprediksi diabetes secara tepat."""