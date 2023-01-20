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

Proyek ini dibangun untuk perusahaan dengan karakteristik bisnis sebagai berikut :

+ Perusahaan memiliki atau membeli rumah dan apartemen kemudian menyewanya ke konsumen.
+ Perusahaan membuka jasa konsultasi harga sewa rumah dan apartemen ke konsumen.

### Problem Statement

1. Fitur apa yang paling berpengaruh terhadap harga sewa rumah atau apartemen?
2. Bagaimana cara memproses data agar dapat dilatih dengan baik oleh model?
3. Berapa harga sewa rumah di pasaran berdasarkan karakteristik tertentu?

### Goals

1. Mengetahui fitur yang paling berpengaruh pada harga sewa rumah atau apartemen.
2. Melakukan persiapan data untuk dapat dilatih oleh model.
3. Membuat model machine learning yang dapat memprediksi harga sewa rumah seakurat mungkin berdasarkan karakteristik tertentu.

### Solution Statement

1. Menganalisis data dengan melakukan univariate analysis dan multivariate analysis. Memahami data juga dapat dilakukan dengan visualisasi. Memahami data dapat membantu untuk mengetahui kolerasi antar fitur dan mendeteksi outlier.
2. Menyiapkan data agar bisa digunakan dalam membangun model.
3. Melakukan hyperparameter tuning menggunakan grid search dan membangun model regresi yang dapat memprediksi bilangan kontinu. ALgoritma yang dipakai dalam proyek ini adalah K-Nearest Neighbour, Random Forest, dan AdaBoost.
