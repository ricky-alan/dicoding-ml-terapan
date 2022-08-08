# Laporan Proyek Machine Learning - Ricky Alan

## Domain Proyek
Diabetes adalah penyakit metabolik yang ditandai dengan kenaikan kadar gula darah yang disebabkan oleh terganggunya fungsi hormon insulin dalam tubuh. Penyakit ini dapat menimbulkan konsekuensi yang serius jika tidak dikontrol dengan baik. Ada dua tipe penyakit diabetes. Diabetes tipe 1 disebabkan karena sistem imun penderita menyerang sel-sel yang memproduksi insulin, sehingga tubuh tidak dapat memproduksi insulin sama sekali. Sedangkan diabetes tipe 2 disebabkan karena tubuh tidak cukup memproduksi insulin atau karena insulin tidak digunakan dengan baik oleh tubuh.

Terdapat beberapa faktor resiko penyakit diabetes, diantaranya karena kelebihan berat badan, kadar kolesterol yang tinggi, gaya hidup tidak aktif atau jarang berolahraga, dan faktor usia. Beberapa penyakit seperti gangguan jantung, gagal ginjal, gangguan penglihatan, gangguan pada tulang dan sendi, serta kerusakan saraf dapat terjadi akibat kadar gula darah yang tinggi. Selain itu, hingga saat ini belum ada obat yang dapat mengobati diabetes secara total. Hal ini tentu menjadi momok bagi semua orang. Oleh karena itu, menerapkan gaya hidup sehat serta deteksi diabetes sejak dini sangat diperlukan untuk dapat mengontrol bahaya penyakit ini.
  
Referensi: [Epidemiology of Diabetes and Diabetes-Related Complications](https://academic.oup.com/ptj/article/88/11/1254/2858146)

## Business Understanding
### Problem Statements
- Faktor apa saja yang berpengaruh terhadap diagnosis diabetes pada seseorang?
- Apakah faktor-faktor yang ada dapat memprediksi diabetes secara tepat?

### Goals
- Mengetahui fitur yang paling berpengaruh terhadap diagnosis diabetes pada seseorang.
- Membuat model machine learning yang dapat memprediksi diabetes secara tepat berdasarkan faktor-faktor yang ada.

### Solution Statements
Untuk mendapatkan model machine learning terbaik untuk prediksi diabetes, saya membandingkan dua algoritma, yaitu K-Nearest Neighbors dan Random Forest. Metriks evaluasi yang saya gunakan adalah accuracy dan mean squared error.

## Data Understanding
Dataset yang saya gunakan merupakan dataset survey 253.680 orang responden yang sudah bersih. Dataset ini terdiri dari 22 kolom (variabel) yang semuanya bertipe data float64. 
Sumber: [Diabetes Health Indicators Dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset?select=diabetes_binary_health_indicators_BRFSS2015.csv).

### Deskripsi Variabel:
- Diabetes_binary merupakan fitur target. Menunjukkan apakah seseorang mengidap diabetes. 0: tidak diabetes, 1: diabetes.
- HighBP: menunjukkan apakah seseorang punya darah tinggi (high blood pressure). 0: tidak, 1: ya.
- HighChol: menunjukkan apakah seseorang punya kolesterol tinggi. 0: tidak, 1: ya.
- CholCheck: menunjukkan apakah seseorang melakukan cek kolesterol dalam 5 tahun. 0: tidak, 1: ya.
- BMI: merupakan kategori berat badan seseorang.
- Smoker: menunjukkan apakah seseorang merokok setidaknya 100 batang rokok selama hidup. 0: tidak, 1: ya.
- Stroke: menunjukkan apakah seseorang mengidap stroke. 0: tidak, 1: ya.
- HeartDiseaseorAttack: menunjukkan apakah seseorang mengidap penyakit jantung. 0: tidak, 1: ya.
- PhysActivity: menunjukkan apakah seseorang melakukan aktivitas fisik (tidak termasuk pekerjaan) dalam 30 hari terakhir. 0: tidak, 1: ya.
- Fruits: menunjukkan apakah seseorang mengonsumsi buah minimal satu kali sehari. 0: tidak, 1: ya.
- Veggies: menunjukkan apakah seseorang mengonsumsi sayur minimal satu kali sehari. 0: tidak, 1: ya.
- HvyAlcoholConsump: merupakan konsumsi alkohol dalam seminggu (pria dewasa >= 14 minuman, wanita dewasa >= 7 minuman).
- AnyHealthcare: menunjukkan apakah seseorang memiliki jaminan kesehatan. 0: tidak, 1: ya.
- NoDocbcCost: menunjukkan apakah dalam 1 tahun terakhir seseorang memiliki kendala biaya saat ingin berobat ke dokter. 0: tidak, 1: ya.
- GenHlth: merupakan kondisi kesehatan secara umum. 1: sangat sangat bagus, 2: sangat bagus, 3: bagus, 4: lumayan, 5: buruk.
- MentHlth: menunjukkan berapa hari seseorang dalam kondisi mental yang buruk (0-30 hari).
- PhysHlth: merupakan luka fisik atau cidera dalam 30 hari terakhir (0-30 hari).
- DiffWalk: menunjukkan apakah seseorang punya kesulitan berjalan atau naik tangga. 0: tidak, 1: ya.
- Sex: jenis kelamin. 0: perempuan, 1: laki-laki.
- Age: kategori umur seseorang. 1: 18-24 , 2: 25-29 , 3 30-34 , 4: 35-39 , 5: 40-44 , 6: 45-49 , 7: 50-54 , 8: 55-59 , 9: 60-64 , 10: 65-69 , 11: 70-74 , 12: 75-79 , 13: >= 80.
- Education: tingkat pendidikan. 1: Tidak Sekolah atau TK, 2: Kelas 1-8, 3: Kelas 9-11, 4: Kelas 12, 5: Perguruan Tinggi 1-3 tahun, 6: Perguruan Tinggi 4 tahun lebih.
- Income: pendapatan. 1: kurang dari \$10,000, 2: kurang dari \$15,000, 3: kurang dari \$20,000, 4: kurang dari \$25,000, 5: kurang dari \$35,000, 6: kurang dari \$50,000, 7: kurang dari \$75,000, 8: \$75,000 atau lebih.

### Univariate Analysis
Salah satu cara untuk melakukan univariate analysis adalah dengan histogram data. Diperoleh hasil sebagai berikut:

![Univariate Analysis](https://github.com/ricky-alan/dicoding-ml-terapan/blob/main/S1_Predictive_Analytic/image/univariate_analysis.png?raw=True)

Dari grafik diatas, dapat diketahui distribusi data setiap variabel.

### Multivariate Analysis
Multivariate Analysis dilakukan untuk mengetahui korelasi / hubungan antar 2 variabel atau lebih. Salah satu caranya adalah dengan menggunakan Matriks Korelasi.

![Multivariate Analysis](https://github.com/ricky-alan/dicoding-ml-terapan/blob/main/S1_Predictive_Analytic/image/multivariate_analysis.png?raw=True)

Dari matriks diatas, dapat diketahui bahwa fitur yang mempunyai korelasi yang kuat dengan fitur target, diantaranya HighBP, HighCol, BMI, HeartDiseaseorAttack, GenHlth, PhyscHlth, DiffWalk, dan Age.

## Data Preparation

### Train Test Split
Sebelum memasuki proses modelling, dataset perlu dibagi terlebih dahulu menjadi training data dan testing data. Proses ini dapat dilakukan dengan mudah menggunakan method train_test_split dari library sklearn.

![Train Test Split](https://github.com/ricky-alan/dicoding-ml-terapan/blob/main/S1_Predictive_Analytic/image/train_test_split.png?raw=True)

dimana:  
X adalah fitur, merupakan faktor-faktor untuk memprediksi diabetes.  
Y adalah label, menetukan diabetes atau tidak diabetes.  
test_size mengatur ukuran testing data. 90% training data, 10% testing data.  
random_state mengatur random generator dalam pembagian dataset.

Pembagian dataset ini dilakukan agar memudahkan kita untuk menguji performa model yang dibangun, seberapa tepat prediksi model terhadap data baru, yaitu testing data. 

## Modeling
Pada tahap ini, saya menggunakan 2 algoritma yaitu K-Nearest Neighbors dan Random Forest sebagai perbandingan.

**Algoritma K-Nearest Neighbors** 

Untuk mengimplementasikan algoritma KNN, saya menggunakan method **KNeighborsClassifier** dari **sklearn.neighbors** dengan argumen n_neighbors=12 yang merupakan banyak tetangga terdekat.

Kelebihan Algoritma KNN:  
- Mudah diimplementasikan
- Efektif terhadap data yang besar
- Kuat dalam training noisy data

Kekurangan Algoritma KNN:
- Perlu menentukan nilai parameter K
- Nilai komputasi yang tinggi
- Rentan terhadap variabel yang non-informatif

**Algoritma Random Forest** 

Untuk mengimplementasikan algoritma Random Forest, saya menggunakan method **RandomForestClassifier** dari **sklearn.ensemble** dengan argumen n_estimators=100 yang menentukan jumlah tree di forest, max_depth=10 yang menentukan kedalaman atau panjang pohon, random_state=55 yang mengontrol random number generator yang digunakan, serta n_jobs=-1 yang berarti semua proses berjalan secara paralel.

Kelebihan Algoritma Random Forest:  
- Dapat mengatasi training data yang besar secara efisien
- Dapat memperkiraan variabel apa yang penting dalam klasifikasi
- Dapat menangani missing values

Kekurangan Algoritma Random Forest:
- Kompleksitas yang tinggi
- Waktu pemrosesan yang lama
- Lebih sulit diimplementasikan

Berdasarkan uraian di atas serta pada saat proses modeling dan evaluasi, menurut saya kedua algoritma bekerja dengan cukup baik dalam memprediksi diabetes. Hal ini dapat dilihat dari nilai akuarasi dan MSE pada saat training dan testing. Namun, algoritma **Random Forest** bekerja lebih baik karena waktu training yang lebih cepat serta tidak terindikasi overfitting.

## Evaluation
Metriks evaluasi yang saya gunakan pada project ini adalah akurasi dan MSE. 

Akurasi diperoleh dari menghitung jumlah prediksi yang benar dibagi dengan total sampel.

![Accuracy](https://github.com/ricky-alan/dicoding-ml-terapan/blob/main/S1_Predictive_Analytic/image/Accuracy.png?raw=True)

Sedangkan MSE atau Mean Squared Error diperoleh dari menghitung jumlah selisih kuadrat rata-rata nilai sebenarnya dengan nilai prediksi.

![MSE](https://github.com/ricky-alan/dicoding-ml-terapan/blob/main/S1_Predictive_Analytic/image/MSE.png?raw=True)

**Algoritma K-Nearest Neighbors**

Training Data:  
- Accuracy Score: 0.8590250137997153
- Mean Squared Error: 0.1409749862002847

Testing Data:
- Accuracy Score: 0.8436029283597699
- Mean Squared Error: 0.1563970716402301

Hasil tersebut menunjukkan bahwa algortima KNN bekerja dengan baik, namun sedikit terindikasi overfitting karena perbedaan score antara training dan testing.

**Algoritma Random Forest**

Training Data:
- Accuracy Score: 0.8552773016472502
- Mean Squared Error: 0.14472269835274978

Testing Data:
- Accuracy Score: 0.852100400906397
- Mean Squared Error: 0.14789959909360292

Hasil tersebut menunjukkan bahwa algortima Random Forest bekerja dengan baik karena tidak terindikasi overfitting.

**Prediksi**

![Prediksi](https://github.com/ricky-alan/dicoding-ml-terapan/blob/main/S1_Predictive_Analytic/image/Prediksi.png?raw=True)

Dari hasil prediksi pada 5 sampel yang ditunjukkan gambar di atas, kedua algoritma dapat memprediksi diabetes dengan tepat.