# Saudi Arabia Used Cars Prediction (Machine Learning) 
****

Repository ini berisi tentang *project* pemodelan *Machine Learning* untuk memprediksi harga mobil bekas di Arab Saudi menggunakan dataset [Saudi Arabia Used Cars](https://www.kaggle.com/datasets/turkibintalib/saudi-arabia-used-cars-dataset?select=UsedCarsSA_Clean_EN.csv). Project ini menghasilkan model *Machine Learning* yang dapat digunakan pada dataset tersebut.<br> <br>
Berikut ada rangkuman dari hasil pengerjaan project ini:

### **Contents**

1. Business Problem Understanding
2. Data Understanding
3. Data Preprocessing
4. Modeling
5. Conclusion
6. Recommendation

****

### ***1. Business Problem Understanding***

***Context***
**Latar Belakang**

Transportasi menjadi alat yang vital dalam menunjang kehidupan manusia. Salah satu jenis transportasi yang paling banyak diminati di Arab Saudi adalah mobil. Mobil menjadi kebutuhan transportasi tingkat primer dibandingkan dengan motor di Arab Saudi. Karena suhu yang tinggi di Arab, membuat kendaraan motor sangat tidak diminati dan membuat kendaraan mobil lebih banyak dicari. Selain hal tadi dengan dihilangkannya aturan wanita yang dilarang untuk bekerja dan mengemudi pada 2019, membuat mobil semakin banyak diminati. Artikel yang memuat tentang *Market Opportunity* mobil bekas di Arab Saudi, mengatakan bahwa rasio membeli mobil bekas dibandingkan mobil baru adalah 2 : 1. Hal ini masih bisa meningkat seiring berjalannya waktu dikarenakan pertumbuhan ekonomi yang baik di Arab Saudi.
<br>
<br>
Karena banyaknya mobil bekas yang diminati, terdapat sejumlah platform pada *website* seperti syarah.com yang tertarik sebagai media jual-beli mobil bekas. syarah.com yang berdiri sejak 2015 menjadi salah satu platform jual-beli mobil bekas terbesar di Arab Saudi. Di syarah.com, pelanggan dapat memilih mobil sesuai keinginan mereka dengan mencocokkan spesifikasi dan harga yang diinginkan. Pada web tersebut juga banyak fitur-fitur yang diperlihatkan sehingga spesifikasi mobil semakin lengkap. Semakin tinggi spesifikasi mobil yang dimiliki, maka semakin tinggi pula harga mobil yang dipasang. JIka berbicara lebih *detail* tentang harga, penentuan harga sebuah mobil bekas akan semakin rumit mengingat sulitnya menentukan harga pasaran dengan melihat spesifikasi yang begitu banyak. Belum lagi munculnya fitur-fitur baru pada mobil terkini yang membuat harga mobil bekas semakin turun. Penentuan harga mobil bekas tidak serta merta dapat dihitung dengan mudah. Oleh sebab itu, dibutuhkannya sebuah *tools* yang bernama *machine learning* untuk membuat harga mobil bekas lebih akurat.
<br>
<br>
*Machine learning* memiliki kemampuan untuk mempelajari berbagai data transaksi sebelumnya dan akan menghasilkan sebuah model yang dapat digunakan oleh perusahaan seperti syarah.com. Dengan adanya *machine learning*, diharapkan harga akan semakin kompetitif di pasaran jual-beli mobil bekas. Alasan lain mengapa dibutuhkannya *tools* ini karena tidak semua calon penjual mengerti tentang spesifikasi yang dimiliki mobil mereka. Jika mereka (penjual) menjual dengan harga yang tinggi, maka mobilnya akan menjadi sulit terjual. Begitu juga sebaliknya jika harga jual terlalu rendah, maka keuntungan yang didapat semakin kecil. Belum lagi syarah.com juga ambil bagian dari hasil transaksi penjualan mobil yang terjadi. Pada akhirnya *machine learning* diharapkan akan membuat seluruh elemen di dunial jual-beli mobil bekas baik itu perusahaan (syarah.com), calon pembeli maupun calon penjual, akan sama-sama diuntungkan dengan diberikannya harga mobil bekas yang akurat dan kompetitif.

![gambar arab](https://media.architecturaldigest.in/wp-content/uploads/2019/05/saudi-saudi-arabia-residency-permanent-residence-expats.jpg)
[Klik Sumber](https://media.architecturaldigest.in/wp-content/uploads/2019/05/saudi-saudi-arabia-residency-permanent-residence-expats.jpg)

***Problem Statement***

Permasalahan akan berdampak langsung kepada perusahaan dan calon penjual dikarenakan bagi calon pembeli, harga yang murah di bawah standar pasaran pasti akan banyak diminati. Bagi perusahaan, pemecahan masalah untuk mendapatkan model bisnis yang baik akan berdampak positif dengan membuat finansial dan *user experience* calon penjual dan pembeli menjadi lebih baik sehingga lebih banyak diminati.
<br>
<br>
Sedangkan bagi calon penjual, model bisnis yang dihasilkan dari perusahaan akan membuat waktu yang digunakan untuk riset harga mobil bekas semakin terpangkas. Tidak lupa harga penjualan akan dijajakan secara kompetitif sehingga keuntungan yang diambil semakin maksimal.
<br>
<br>
Bagi calon pembeli, waktu yang mereka gunakan untuk membandingkan harga juga akan berkurang. Hal ini disebabkan jika model bisnis perusahaan dapat dihasilkan, maka seluruh harga yang ada di platform tersebut adalah harga yang kompetitif sehingga tidak perlu khawatir harga yang mereka lihat apakah terlalu mahal atau terlalu murah.
<br>
<br>
Dengan banyaknya berbagai spesifikasi dan kondisi pada mobil bekas, **perusahaan diharapkan mampu memberikan model bisnis yang baik dalam menentukan harga yang akurat** supaya banyak calon pembeli atau pelanggan yang semakin tertarik pada perusahaan ini. Calon pembeli juga akan diberikan kemudahan dalam menentukan spesifikasi dan harga yang diinginkan tanpa khawatir apakah pilihannya salah atau sudah tepat. Harga yang akurat dan kemudahan inilah yang dapat menarik minat banyak pelanggan untuk belanja di perusahaan ini dan tentu saja akan meningkatkan pendapatan finansial perusahaan.

***Goals***

Berdasarkan permasalahan tersebut, perusahaan syarah.com perlu memiliki *tools* supaya calon pembeli dapat menentukan mobil impian mereka dengan melihat berbagai spesifikasi dan harga yang akurat sesuai keinginan dan kebutuhan mereka. Variabel-variabel spesifikasi mobil seperti merk mobil, tahun mobil, jarak tempuh dan lain-lain diharapkan mampu meningkatkan akurasi yang baik dalam menentukan harga mobil. Hal inilah yang membuat harga mobil mampu bersaing di pasaran sehingga perusahaanpun dapat mengambil keuntungan semaksimal mungkin.

***Analytic Approach***

Hal pertama yang harus dilakukan adalah dengan menganalisis seluruh data supaya dapat menemukan pola dari fitur-fitur yang diberikan dan yang membedakan antara satu mobil dengan yang lainnya. Langkah selanjutnya akan dibuat sebuah model regresi yang akan membantu perusahaan untuk dapat menyediakan *tool* prediksi harga mobil bekas. Penentuan model regresi akan ditentukan dengan melihat matrix evaluasi terbaik sehingga *final model* dari *machine learning* dapat ditentukan.

***Metric Evaluation***

Pada model yang akan dibuat saat pembersihan data *outliers*, hanya data *outliers* yang bernilai ekstrem saja yang akan dihapus. Oleh karena itu, data tetap akan memiliki sejumlah *outliers* yang didiamkan. Hal ini membuat pada model ini *metric evaluation* yang akan digunakan adalah *metric-metric* yang tidak sensitif terhadap *outliers*. *Metric* seperti MSE, RMSE, dan RMSPE tidak cocok digunakan pada data ini karena sangat sensitif terhadap data *outliers*. *Metric evaluation* yang akan digunakan adalah:

1. R-Square:
2. MAE (Mean Absolute Error)
3. MAPE (Mean Absolute Percentage Error)
4. RMSLE (Root Mean Squared Log Error)

### ***2. Data Understanding***

- Deskripsi dari tiap kolom didapat berdasarkan *domain knowledge* yang dipelajari.
- Dataset merupakan data daftar harga mobil dari syarah.com di Saudi Arabia.
- Setiap baris data merepresentasikan informasi terkait informasi spesifikasi mobil hingga harga mobil.

**Attributes Information**

| **Attribute** | **Data Type** | **Description** |
| --- | --- | --- |
| Type | Object | Merek mobil |
| Region | Object | Daerah penjualan mobil bekas |
| Make | Object | Nama perusahaan pembuat mobil |
| Gear_Type | Object | Tipe gear yang digunakan (Automatic / Manual) |
| Origin | Object | Negara pengimpor mobil (Gulf / Saudi / Other / Unknown) |
| Options | Object | Option used (Full Options / Semi-Full / Standard) |
| Year | Integer | Tahun pembuatan mobil |
| Engine_Size | Float | Ukuran mesin mobil |
| Mileage | Integer | Jarak yang sudah ditempuh kendaraan (KM) |
| Negotiable | Boolean | Jika True, maka harga adalah 0 karena harga ditentukan lewat negosiasi |
| Price | Integer | Harga mobil bekas (SAR) |

<br>

"Price" akan menjadi label atau *target* kolom yang akan diprediksi (*dependent variable*), kolom lain akan menjadi *predictor* variabel / *independent variable* yang akan memprediksi "Price".

![Datasets](https://user-images.githubusercontent.com/107677479/188270650-66c9d222-015e-4f66-9d10-ce6d882f3937.PNG)

### ***3. Data Preprocessing***

Pada tahap ini akan dihapus data-data yang duplikat dan *features* yang tidak relevan yaitu kolom "Negotiable" dan kolom "Origin". Selain dual hal tadi, akan dihapus juga data-data yhang sifatnya adalah *extreme outliers*.

![image](https://user-images.githubusercontent.com/107677479/188270904-435f8db7-70a6-43fb-8b8b-7fa207869abf.png)
