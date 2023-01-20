# Proyek Pertama Prediction Housing Price in Paris 

#### Disusun oleh : Ayomi Sasmito

Proyek pertama predictive analysis dicoding. Proyek ini membuat model machine learning untuk memprediksi harga rumah di paris

## Domain Proyek

### Latar Belakang

Rumah adalah tempat untuk melepaskan lelah, tempat bergaul, dan membina rasa kekeluargaan diantara anggota keluarga, tempat berlindung keluarga dan menyimpan barang berharga, dan rumah juga sebagai status lambang sosial (Azwar, 1996; Mukono, 2000).

<br>

<div><img src="https://d3p0bla3numw14.cloudfront.net/news-content/img/2021/05/21041605/rumah-idaman-minimalis.jpg" width="600"/></div>

[Referensi gambar](https://rumah123.com)

<br>
Setiap rumah memiliki harga yang bervariasi berdasarkan fitur yang ditawarkan didalamnya. Harga yang bervariasi tersebut menyebabkan ketidakpastian yang tinggi dan sulit jika memprediksi harga rumah secara akurat di masa depan. Ketidakpastian ini dapat diminimalisir dengan membuat sebuah sistem prediksi yang dapat menentukan berapa harga yang layak diberikan untuk karakteristik rumah tertentu. 

Untuk mencapai hal tersebut, perlu dilakukan sebuah prediksi terhadap harga sewa rumah atau yang dikenal dengan housing dengan menggunakan machine learning. Diharapkan dengan model ini , harga sewa mampu diprediksi sesuai dengan harga pasar. Prediksi ini juga akan menjadi saran untuk para developer dalam menyewa rumah dengan harga yang dapat mendatangkan profit bagi mereka. Di dalam model ini nantinya akan disimulasikan menggunakan dataset 

Referensi : [Algoritma K-Nearest Neighbour untuk Memprediksi Harga Jual Tanah](https://journal.unhas.ac.id/index.php/jmsk/article/download/3399/1936/6761)

## Business Understanding

Proyek ini dibangun untuk developer dengan karakteristik bisnis sebagai berikut :

+ Developer di paris memiliki sejumlah rumah kemudian menyewakannya ke calon pelanggan.
+ Developer di paris membuka jasa konsultasi harga sewa rumah kepada para calon pelanggan.

### Problem Statement

1. Fitur apa yang paling berpengaruh terhadap harga sewa rumah di paris?
2. Bagaimana cara memproses data agar dapat dilatih dengan baik oleh model?
3. Berapa harga sewa rumah di pasaran berdasarkan karakteristik tertentu?

### Goals

1. Mengetahui fitur yang paling berpengaruh pada harga sewa rumah di paris.
2. Melakukan persiapan data untuk dapat dilatih oleh model.
3. Membuat model machine learning yang dapat memprediksi harga sewa rumah seakurat mungkin berdasarkan karakteristik tertentu.

### Solution Statement

1. Analisis data menggunakan analisis univariat dan analisis multivariat. Memahami informasi juga dapat dilakukan dengan bantuan visualisasi. Memahami data dapat membantu menemukan korelasi antara karakteristik dan mengidentifikasi outlier.
2. Persiapkan data yang akan digunakan untuk membuat model.
3. Lakukan penyetelan hyperparameter menggunakan pencarian grid dan buat model regresi yang dapat memprediksi angka kontinu. Algoritma yang digunakan dalam proyek ini adalah K-Nearest Neighbor, Random Forest dan AdaBoost.

## Data Understanding & Removing Outlier

Dataset yang digunakan dalam proyek ini merupakan data harga sewa rumah dengan berbagai karakteristik di India. Dataset ini dapat diunduh di [Kaggle : Paris Housing Dataset](https://www.kaggle.com/datasets/mssmartypants/paris-housing-price-prediction?select=ParisHousing.csv).

Berikut informasi pada dataset :

+ Dataset memiliki format CSV (Comma-Seperated Values).
+ Dataset memiliki 10000 sample dengan 17 fitur.
+ Dataset memiliki 17 fitur bertipe int64 dan 1 fitur bertipe float.
+ Tidak ada missing value dalam dataset.

### Variable - variable pada dataset
+ Square Meters : Luas Rumah.
+ NumberOfRooms : Banyaknya ruangan.
+ hasYard : Status adanya halaman.
+ hasPool : Status adanya kolam renang. 
+ Floors : Jumlah lantai dalam rumah. 
+ cityCode : Jumlah Kota.
+ cityPartRange : Semakin tinggi nilainya, semakin ekslusif lingkungan rumah tersebut.
+ numPrevOwners - number of prevoious owners
+ made year : tahun dibuat.
+ isNewBuilt : Apakah bangunan baru
+ hasStormProtector : Status adanya pelindung dari petir pada bangunan rumah
+ basement : Luas basement.
+ attic : Luas loteng.
+ garage : Ukuran garasi.
+ hasStorageRoom : Status adanya gudang.
+ hasGuestRoom : Status adanya ruang tamu.
+ price : Harga.

Dari ke 12 fitur dapat dilihat bahwa fitur Point of Contract dan Posted On tidak mempengaruhi harga sewa rumah sehingga akan dihapus. Hal ini dikarenakan kedua fitur tersebut tidak diperlukan dalam membangun model prediksi harga sewa.

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

