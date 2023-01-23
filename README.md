# Proyek Pertama Weather Prediction
#### Disusun oleh : Ayomi Sasmito

Proyek pertama predictive analysis dicoding. Proyek ini membuat model machine learning untuk memprediksicuaca berdasarkan data yang diberikan.

## Domain Proyek

### Latar Belakang

Cuaca merupakan keadaan udara pada saat tertentu dan di wilayah tertentu yang relatif sempit pada jangka waktu yang singkat. Cuaca terbentuk dari gabungan unsur cuaca dan jangka waktu cuaca bisa hanya beberapa jam saja. Misalnya pagi hari, siang hari, sore hari atau malam hari dan keadaannya bisa berbeda-beda untuk setiap tempat serta setiap jamnya. Di Indonesia keadaan cuaca selalu diumumkan untuk jangka waktu sekitar 24 jam melalui prediksi cuaca yang dikembangkan oleh Badan Meteorologi Klimatologi dan Geofisika (BMKG), Departemen Perhubungan. Selain di Indonesia , dinegara lain memiliki lembaga khusus untuk memberikan prediksi cuaca yang akan terjadi dalam 24 jam. Kemungkinan cuaca yang akan terjadi adalah berawan, berkabut, hujan, berawan, salju. 

Dalam penelitian ini, akan disusun sebuah model machine learning yang bertujuan untuk memprediksi cuaca yang akan terjadi dan melakukan perbandingan antara model-model yang ada. Model yang dibuat nantinya akan di evaluasi dengan menggunakan Mean Square Error (MSE) dan kemudian akan dipilih manakah MSE terkecil yang menunjukkan model terbaik yang akan digunakan dalam machine learning.

<br>

<div><img src="https://user-images.githubusercontent.com/40420367/213976580-053f6395-463a-4bb0-b878-997365fa7498.jpg" width="500"/></div>

[Referensi gambar](https://palopopos.fajar.co.id/2022/07/11/cuaca-palopo-hari-ini-dan-besok-gerimis-diiringi-badai-petir/)

<br>


Referensi : [Prediksi Cuaca Kota Denpasar menggunakan Algoritma ELM dengan Optimasi Quantum Delta Particle Swarm Optimization](https://j-ptiik.ub.ac.id/index.php/j-ptiik/article/download/8746/4018/)

### Problem Statement

1. Fitur apa yang paling berpengaruh terhadap cuaca pada hari ini ?
2. Bagaimana cara memproses data agar dapat dilatih dengan baik oleh model?

### Goals

1. Mengetahui fitur yang paling berpengaruh pada prediksi cuaca.
2. Melakukan persiapan data untuk dapat dilatih oleh model.
3. Membuat model machine learning yang dapat memprediksi cuaca pada kondisi tertentu.

### Solution Statement

1. Analisis data menggunakan analisis univariat dan analisis multivariat. Memahami informasi juga dapat dilakukan dengan bantuan visualisasi. Memahami data dapat membantu menemukan korelasi antara karakteristik dan mengidentifikasi outlier.
2. Persiapkan data yang akan digunakan untuk membuat model.
3. Lakukan penyetelan hyperparameter menggunakan pencarian grid dan buat model regresi yang dapat memprediksi angka kontinu. Algoritma yang digunakan dalam proyek ini adalah K-Nearest Neighbor, Random Forest dan AdaBoost.

## Data Understanding & Removing Outlier

Dataset yang digunakan dalam proyek ini merupakan data cuaca. Dataset ini dapat diunduh di [Kaggle : Weather Prediction](https://www.kaggle.com/datasets/ananthr1/weather-prediction).

Berikut informasi pada dataset :

+ Dataset memiliki format CSV (Comma-Seperated Values).
+ Dataset memiliki 1461 sample dengan 6 fitur.
+ Dataset memiliki 4 fitur bertipe int64 dan 2 fitur bertipe object.
+ Tidak ada missing value dalam dataset.

### Variable - variable pada dataset
+ date - tanggal
+ precipitation - All forms in which water falls on the land surface and open water bodies as rain, sleet, snow, hail, or drizzle.
+ temp_max - Maximum Temperature
+ temp_min - Minimum Temperature
+ wind - wind speed
+ weather - output

Dari ke 6 fitur dapat dilihat bahwa fitur date tidak mempengaruhi harga sewa rumah sehingga akan dihapus. Hal ini dikarenakan kedua fitur tersebut tidak diperlukan dalam membangun model prediksi cuaca.

### Univariate Analysis

Univariate Analysis adalah menganalisis setiap fitur secara terpisah.

#### Analisis sebaran pada setiap fitur numerik

<div><img src="https://user-images.githubusercontent.com/40420367/213976242-2bf41610-ffce-4c4f-b4b5-16f26f57ee5b.png" width="450"/></div><br />
Berikut analisis dari grafik di atas :

+ Curah hujan / precipitation banyak yang berada dakam range 0-1.
+ Temperatur maksimum rata-rata adalah 16.3 derajat celcius.
+ Temperatur minimum rata-rata adalah 8.2 derajat celcius.
+ Kecepatan angin rata-rata adalah 3.2 

### Multivariate Analysis

Multivariate Analysis menunjukkan hubungan antara dua atau lebih fitur dalam data.

#### Analisis numerik
  
+ Melihat kolerasi antara semua fitur numerik
  <div><img src="https://user-images.githubusercontent.com/40420367/213976380-fa4f7288-be1f-4b1f-8f16-dff0684419b2.png" width="450"/></div>
  Fitur temp_max, temp_min, precipitation, wind berkorelasi tidak signifikan dengan fitur target weather. Hal ini mungkin   disebabkan oleh kurangnya data dalam penelitian ini.Fitur temp_min dan temp_max berkorelasi dengan weather. 

#### Data preparation

+ Train Test Split

Train test split aja proses membagi data menjadi data latih dan data uji. Data latih akan digunakan untuk membangun model, sedangkan data uji akan digunakan untuk menguji performa model. Pada proyek ini dataset sebesar 1461 dibagi menjadi 1314 untuk data latih dan 147 untuk data uji.

+ Normalization

Algoritma machine learning akan memiliki performa lebih baik dan bekerja lebih cepat jika dimodelkan dengan data seragam yang memiliki skala relatif sama. Salah satu teknik normalisasi yang digunakan pada proyek ini adalah Standarisasi dengan sklearn.preprocessing.StandardScaler.

#### Modeling
Algoritma Penelitian ini melakukan pemodelan dengan 3 algoritma, yaitu K-Nearest Neighbour, Random Forest, dan Boosting

+ Algoritma
  Penelitian ini melakukan pemodelan dengan 3 algoritma, yaitu K-Nearest Neighbour, Random Forest, dan
  + K-Nearest Neighbour
    K-Nearest Neighbour bekerja dengan membandingkan jarak satu sampel ke sampel pelatihan lain dengan memilih sejumlah k tetangga terdekat. Proyek ini menggunakan [sklearn.neighbors.KNeighborsRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html) dengan memasukkan X_train dan y_train dalam membangun model. Parameter yang digunakan pada proyek ini adalah :
    + `n_neighbors` = Jumlah k tetangga tedekat.

  + Random Forest
    Algoritma random forest adalah teknik dalam machine learning dengan metode ensemble. Teknik ini beroperasi dengan membangun banyak decision tree pada waktu pelatihan. Proyek ini menggunakan [sklearn.ensemble.RandomForestRegressor](https://scikit -learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html) dengan memasukkan X_train dan y_train dalam membangun model. Parameter yang digunakan pada proyek ini adalah :
    + `n_estimators` = Jumlah maksimum estimator di mana boosting dihentikan.
    + `max_depth` = Kedalaman maksimum setiap tree.
    + `random_state` = Mengontrol seed acak yang diberikan pada setiap base_estimator pada setiap iterasi boosting.

  + Adaboost
    AdaBoost juga disebut Adaptive Boosting adalah teknik dalam machine learning dengan metode ensemble.  Algoritma yang paling umum digunakan dengan AdaBoost adalah pohon keputusan (decision trees) satu tingkat yang berarti memiliki pohon Keputusan dengan hanya 1 split. Pohon-pohon ini juga disebut Decision Stumps. Algoritma ini bertujuan untuk meningkatkan performa atau akurasi prediksi dengan cara menggabungkan beberapa model sederhana dan dianggap lemah (weak learners) secara berurutan sehingga membentuk suatu model yang kuat (strong ensemble learner). Proyek ini menggunakan [sklearn.ensemble.AdaBoostRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html) dengan memasukkan X_train dan y_train dalam membangun model. Parameter yang digunakan pada proyek ini adalah :
    + `n_estimators` = Jumlah maksimum estimator di mana boosting dihentikan.
    + `learning_rate` = Learning rate memperkuat kontribusi setiap regressor.
    + `random_state` = Mengontrol seed acak yang diberikan pada setiap base_estimator pada setiap iterasi boosting.

+ Hyperparameter Tuning (Grid Search)
  Hyperparameter tuning adalah cara untuk mendapatkan parameter terbaik dari algoritma dalam membangun model. Salah satu teknik dalam hyperparameter tuning yang digunakan dalam proyek ini adalah grid search. Berikut adalah hasil dari Grid Search pada proyek ini :
  | model    | best_params                                                     |
  |----------|-----------------------------------------------------------------|
  | knn      | {'n_neighbors': 50}                                              |
  | boosting | {'learning_rate': 0.05, 'n_estimators': 55, 'random_state': 55} |
  | rf       | {'max_depth': 16, 'n_estimators': 50, 'random_state': 55}        |
  
#### Evaluation

Metrik evaluasi yang digunakan pada proyek ini adalah akurasi dan mean squared error (MSE). Akurasi menentukan tingkat kemiripan antara hasil prediksi dengan nilai yang sebenarnya (y_test). Mean squared error (MSE) mengukur error dalam model statistik dengan cara menghitung rata-rata error dari kuadrat hasil aktual dikurang hasil prediksi. Berikut formulan MSE :
<div><img src="https://user-images.githubusercontent.com/107544829/188412654-f5dc0ae1-901b-470e-aae5-1f6b5fb68b4d.png](https://www.dicoding.com/academies/319/tutorials/18595#)" width="300"/></div>

Berikut hasil evaluasi pada proyek ini :

+ Akurasi
  | model    | accuracy |
  |----------|----------|
  | knn      | 0.726775 |
  | boosting | 0.898556 |
  | rf       | 0.932057 |

+ Mean Squared Error (MSE)
  <div><img src="https://user-images.githubusercontent.com/107544829/188413846-7d5454b5-7f83-488e-836f-4f3593eb3d5d.png" width="300"/></div>

Dari hasil evaluasi dapat dilihat bahwa model dengan algoritma Random Forest memiliki akurasi lebih tinggi tinggi dan tingkat error lebih kecil dibandingkan algoritma lainnya dalam proyek ini.
