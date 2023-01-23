# Proyek Pertama Weather Prediction
#### Disusun oleh : Ayomi Sasmito

Proyek pertama predictive analysis dicoding. Proyek ini membuat model machine learning untuk memprediksicuaca berdasarkan data yang diberikan.

## Domain Proyek

### Latar Belakang

Cuaca merupakan keadaan udara pada saat tertentu dan di wilayah tertentu yang relatif sempit pada jangka waktu yang singkat. Cuaca terbentuk dari gabungan unsur cuaca dan jangka waktu cuaca bisa hanya beberapa jam saja. Misalnya pagi hari, siang hari, sore hari atau malam hari dan keadaannya bisa berbeda-beda untuk setiap tempat serta setiap jamnya. Di Indonesia keadaan cuaca selalu diumumkan untuk jangka waktu sekitar 24 jam melalui prediksi cuaca yang dikembangkan oleh Badan Meteorologi Klimatologi dan Geofisika (BMKG), Departemen Perhubungan. Selain di Indonesia , dinegara lain memiliki lembaga khusus untuk memberikan prediksi cuaca yang akan terjadi dalam 24 jam. Kemungkinan cuaca yang akan terjadi adalah berawan, berkabut, hujan, berawan, salju. 

Dalam penelitian ini, akan disusun sebuah model machine learning yang bertujuan untuk memprediksi cuaca yang akan terjadi dan melakukan perbandingan antara model-model yang ada. Model yang dibuat nantinya akan di evaluasi dengan menggunakan Mean Square Error (MSE) dan kemudian akan dipilih manakah MSE terkecil yang menunjukkan model terbaik yang akan digunakan dalam machine learning.

<br>

<div><img src="(https://d3p0bla3numw14.cloudfront.net/news-content/img/2021/05/21041605/rumah-idaman-minimalis.jpg)" width="600"/></div>

[Referensi gambar]([https://rumah123.com](https://palopopos.fajar.co.id/2022/07/11/cuaca-palopo-hari-ini-dan-besok-gerimis-diiringi-badai-petir/))

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

Dataset yang digunakan dalam proyek ini merupakan data harga sewa rumah dengan berbagai karakteristik di India. Dataset ini dapat diunduh di [Kaggle : Weather Prediction](https://www.kaggle.com/datasets/ananthr1/weather-prediction).

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

<div><img src="(https://user-images.githubusercontent.com/40420367/213975355-faab7c51-e852-460a-af03-ba2ddb095a21.png)" width="450"/></div><br />
Berikut analisis dari grafik di atas :

+ Curah hujan / precipitation banyak yang berada dakam range 0-1.
+ Temperatur maksimum rata-rata adalah 16.3 derajat celcius.
+ Temperatur minimum rata-rata adalah 8.2 derajat celcius.
+ Kecepatan angin rata-rata adalah 3.2 

### Multivariate Analysis

Multivariate Analysis menunjukkan hubungan antara dua atau lebih fitur dalam data.

#### Analisis fitur numerik
  
+ Melihat kolerasi antara semua fitur numerik
  <div><img src="https://user-images.githubusercontent.com/107544829/188323797-8186246a-8cdd-4232-8bc7-bce615cf92d0.png" width="350"/></div>
  Fitur BHK, Size, dan Bathroom berkorelasi tidak signifikan dengan fitur target (Rent). Hal ini mungkin   disebabkan oleh kurangnya data dalam penelitian ini.Fitur BHK dan Bathroom berkolerasi signifikan dengan fitur size. Hal ini sudah sesuai harapan dari penghapusan outlier yang sudah dilakukan sebelumnya.

