# Proyek Pertama Weather Prediction
#### Disusun oleh : Ayomi Sasmito

Proyek pertama predictive analysis dicoding. Proyek ini membuat model machine learning untuk memprediksicuaca berdasarkan data yang diberikan.

## Domain Proyek

### Latar Belakang

Cuaca merupakan keadaan udara pada saat tertentu dan di wilayah tertentu yang relatif sempit pada jangka waktu yang singkat. Cuaca terbentuk dari gabungan unsur cuaca dan jangka waktu cuaca bisa hanya beberapa jam saja. Misalnya pagi hari, siang hari, sore hari atau malam hari dan keadaannya bisa berbeda-beda untuk setiap tempat serta setiap jamnya. Di Indonesia keadaan cuaca selalu diumumkan untuk jangka waktu sekitar 24 jam melalui prediksi cuaca yang dikembangkan oleh Badan Meteorologi Klimatologi dan Geofisika (BMKG), Departemen Perhubungan. Selain di Indonesia , dinegara lain memiliki lembaga khusus untuk memberikan prediksi cuaca yang akan terjadi dalam 24 jam. Kemungkinan cuaca yang akan terjadi adalah berawan, berkabut, hujan, berawan, salju. 

Dalam penelitian ini, akan disusun sebuah model machine learning yang bertujuan untuk memprediksi cuaca yang akan terjadi dan melakukan perbandingan antara model-model yang ada. Model yang dibuat nantinya akan di evaluasi dengan menggunakan Mean Square Error (MSE) dan kemudian akan dipilih manakah MSE terkecil yang menunjukkan model terbaik yang akan digunakan dalam machine learning.

<br>

<div><img src="https://d3p0bla3numw14.cloudfront.net/news-content/img/2021/05/21041605/rumah-idaman-minimalis.jpg" width="600"/></div>

[Referensi gambar](https://rumah123.com)

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

#### Analisis jumlah nilai unique pada setiap fitur kategorik

Fitur kategorik City, Furnishing Status, dan Tenant Preferred memiliki sebaran sample yang cukup merata.
<div><img src="https://user-images.githubusercontent.com/107544829/188319357-fc12fffa-b709-4584-8363-778bc678b328.png" width="220"/></div> <div><img src="https://user-images.githubusercontent.com/107544829/188319651-02ddb783-da3d-41ed-9b5f-9525aaaf9ed1.png" width="220"/></div> <div><img src="https://user-images.githubusercontent.com/107544829/188319750-1f080942-7826-4eaf-a021-8b9f938a861a.png" width="220"/></div><br />

Berikut adalah fitur dengan sample yang tidak merata :

+ Area Type
  <div><img src="https://user-images.githubusercontent.com/107544829/188318629-f474b626-a16a-4971-ab42-2c183d22b744.png" width="220"/></div>
  Hanya terdapat 2 sample Built Area pada fitur Area Type. Untuk menghindari high dimensional data, maka kedua sample ini akan dihapus.

+ Floor dan Area Locality
  <div><img src="https://user-images.githubusercontent.com/107544829/188319871-603b24b8-26b2-449b-b42e-59501a4803a7.png" width="220"/></div>
   <div><img src="https://user-images.githubusercontent.com/107544829/188319880-3226bd04-920e-4050-b5ab-38dec02fc524.png" width="220"/></div>
  Fitur Floor dan Area Locality memiliki banyak sekali nilai unique. Untuk menghindari high dimensional data, maka kedua fitur ini akan dihapus.

#### Analisis sebaran pada setiap fitur numerik

<div><img src="https://user-images.githubusercontent.com/107544829/188320722-451f25bd-de65-4e09-9d0a-9d8835249492.png" width="450"/></div><br />
Berikut analisis dari grafik di atas :

+ Sebagian besar rumah memiliki 1 sampai 3 BHK dan 1 sampai 3 kamar mandi.
+ Sebagian besar rumah memiliki luas di bawah 2000 sqft.
+ Rentang harga sewa cukup tinggi, yaitu dari 1200 hingga 3500000. Namun, rata-rata harga rumah hanya 35003. Distribusi harga yang kurang bagus seperti ini dapat berimplikasi pada model.

### Multivariate Analysis

Multivariate Analysis menunjukkan hubungan antara dua atau lebih fitur dalam data.

#### Analisis fitur numerik

+ Fitur Size dan BHK (Menghapus BHK Outlier)
  Kedua fitur ini dianalisis karena tidak biasa untuk rumah dengan 1 BHK memiliki luas 100 sqft. Untuk itu ditentukan treshold atau batas 300 sqft/bhk. Data yang berada di bawah batas akan dihapus. Hal ini menyebabkan berkurangnya jumlah sample sebesar 548.

+ Fitur Size dan Rent (Menghapus Price per sqft Outlier)
  Untuk memudahkan dalam mendeteksi outlier, maka dibuat fitur baru 'Price_per_sqft' dari kedua fitur tersebut untuk menganalisis harga sewa per luas sqft.
  <div><img src="https://user-images.githubusercontent.com/107544829/188323140-6174b592-4c7b-4671-9acb-b49a621d2aba.png" width="220"/></div>
  Dari sini dapat terlihat bahwa harga 571 per sqft sangat rendah dan harga 1400000 per sqft sangat tinggi. Untuk itu penghapusan outlier price per sqft outlier dengan mean dan one standard deviation yang telah dikelompokkan berdasarkan kota. Hal ini menyebabkan berkurangnya jumlah sample sebesar 497.

+ Fitur Bathroom dan BHK (Menghapus Bathroom Outlier)
  Kedua fitur ini dianalisis karena tidak biasa untuk rumah dengan 2 BHK memiliki 4 kamar mandi. Untuk itu ditentukan batas bahwa jumlah kamar mandi tidak boleh melebihi jumlah BHK + 2. Hal ini menyebabkan berkurangnya sample sebesar 3.
  
+ Melihat kolerasi antara semua fitur numerik
  <div><img src="https://user-images.githubusercontent.com/107544829/188323797-8186246a-8cdd-4232-8bc7-bce615cf92d0.png" width="350"/></div>
  Fitur BHK, Size, dan Bathroom berkorelasi tidak signifikan dengan fitur target (Rent). Hal ini mungkin   disebabkan oleh kurangnya data dalam penelitian ini.Fitur BHK dan Bathroom berkolerasi signifikan dengan fitur size. Hal ini sudah sesuai harapan dari penghapusan outlier yang sudah dilakukan sebelumnya.

