# PREDIKSI GAJI BERDASARKAN LAMA PENGALAMAN KERJA MENGGUNAKAN SUPPORT VECTOR REGRESSION (SVR)

## 1. Pendahuluan

Perkembangan teknologi Machine Learning memungkinkan proses analisis data dilakukan untuk menghasilkan prediksi berdasarkan pola yang terdapat dalam data historis. Salah satu permasalahan yang dapat diselesaikan menggunakan pendekatan Machine Learning adalah memperkirakan nilai gaji berdasarkan lama pengalaman kerja.

Gaji merupakan salah satu aspek yang dapat dipengaruhi oleh pengalaman kerja seseorang. Semakin lama seseorang memiliki pengalaman kerja, terdapat kemungkinan perubahan nilai gaji yang diterima. Namun, hubungan antara pengalaman kerja dan gaji tidak selalu membentuk pola linear. Oleh karena itu, diperlukan metode regresi yang mampu mempelajari hubungan antara variabel masukan dan nilai keluaran, termasuk hubungan yang bersifat nonlinier.

Pada proyek ini, algoritma yang digunakan adalah **Support Vector Regression (SVR)**. SVR merupakan pengembangan dari Support Vector Machine (SVM) yang digunakan untuk menyelesaikan permasalahan regresi atau memprediksi nilai kontinu. Penelitian Adam et al. (2022) menunjukkan penerapan SVR dengan kernel Radial Basis Function (RBF) untuk melakukan prediksi keluaran daya fotovoltaik. Penelitian tersebut juga membandingkan beberapa kernel untuk memperoleh hasil prediksi yang sesuai.

Proyek ini menggunakan dataset `Salary_Data.csv` untuk membangun model prediksi gaji berdasarkan lama pengalaman kerja. Proses penelitian meliputi pengolahan data, pembagian data latih dan data uji, normalisasi menggunakan StandardScaler, pembangunan model SVR, evaluasi menggunakan MAE, RMSE, dan R², serta visualisasi hasil prediksi.

## 2. Tujuan

Tujuan dari proyek ini adalah:

1. Memahami penerapan algoritma Support Vector Regression untuk permasalahan regresi.
2. Membangun model Machine Learning untuk memprediksi gaji berdasarkan lama pengalaman kerja.
3. Menerapkan proses preprocessing menggunakan StandardScaler pada fitur dan target.
4. Mengukur performa model menggunakan Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), dan R-squared (R²).
5. Menampilkan perbandingan antara data aktual dan hasil prediksi melalui visualisasi.
6. Melakukan pengujian prediksi gaji berdasarkan input pengalaman kerja baru.

## 3. Landasan Teori

### 3.1 Machine Learning

Machine Learning merupakan bagian dari kecerdasan buatan yang memungkinkan komputer mempelajari pola dari sekumpulan data untuk menghasilkan prediksi atau keputusan tanpa harus diberikan aturan secara manual untuk setiap kondisi.

Dalam proyek ini, Machine Learning digunakan untuk mempelajari hubungan antara lama pengalaman kerja sebagai variabel independen dan gaji sebagai variabel dependen. Model dilatih menggunakan data historis sehingga dapat memperkirakan nilai gaji berdasarkan pengalaman kerja yang diberikan.

### 3.2 Support Vector Regression (SVR)

Support Vector Regression merupakan metode regresi yang dikembangkan dari konsep Support Vector Machine (SVM). Berbeda dengan SVM yang umumnya digunakan untuk klasifikasi, SVR digunakan untuk memperkirakan nilai numerik atau kontinu.

SVR bekerja dengan membentuk fungsi regresi yang berusaha menghasilkan prediksi sedekat mungkin dengan nilai aktual. Metode ini menggunakan konsep epsilon-insensitive loss, yaitu memberikan toleransi terhadap kesalahan prediksi selama kesalahan tersebut masih berada dalam batas epsilon yang telah ditentukan.

SVR juga menggunakan fungsi kernel untuk menangani hubungan data yang bersifat nonlinier. Dalam proyek ini, digunakan kernel Radial Basis Function (RBF) agar model dapat mempelajari pola hubungan antara pengalaman kerja dan gaji yang tidak harus berbentuk garis lurus.

### 3.3 Kernel Radial Basis Function (RBF)

Radial Basis Function merupakan salah satu jenis kernel yang digunakan dalam algoritma SVR untuk menangani pola hubungan nonlinier. Kernel RBF menghitung tingkat kemiripan antara dua titik data berdasarkan jarak di antara keduanya.

Penggunaan kernel RBF memungkinkan model mempelajari pola yang lebih fleksibel dibandingkan kernel linear. Pemilihan kernel ini juga digunakan dalam penelitian Adam et al. (2022) dan Maulana et al. (2025) untuk menguji performa model SVR pada permasalahan prediksi.

Pada implementasi ini, kernel RBF ditentukan melalui parameter:

- `kernel="rbf"`: Menggunakan kernel Radial Basis Function.
- `C=100`: Mengatur tingkat penalti terhadap kesalahan prediksi.
- `epsilon=0.1`: Menentukan batas toleransi kesalahan prediksi.
- `gamma="scale"`: Menentukan nilai gamma berdasarkan jumlah fitur dan varians data latih.

### 3.4 StandardScaler

StandardScaler merupakan teknik standardisasi data yang digunakan untuk mengubah distribusi nilai fitur agar memiliki rata-rata mendekati 0 dan standar deviasi mendekati 1.

Standardisasi diperlukan karena algoritma SVR cukup sensitif terhadap perbedaan skala nilai antarvariabel. Dalam proyek ini, standardisasi diterapkan pada variabel pengalaman kerja (X) dan variabel gaji (y).

Proses standardisasi dilakukan menggunakan data latih sebagai dasar perhitungan parameter scaler. Selanjutnya, parameter yang sama diterapkan pada data uji. Cara ini dilakukan untuk menghindari data leakage atau kebocoran informasi dari data pengujian ke dalam proses pelatihan model.

### 3.5 Evaluasi Model

Evaluasi dilakukan untuk mengetahui seberapa baik model menghasilkan prediksi dibandingkan dengan nilai aktual. Terdapat tiga metrik evaluasi yang digunakan dalam proyek ini.

**a. Mean Absolute Error (MAE)**

MAE merupakan metrik yang menghitung rata-rata selisih absolut antara nilai aktual dan nilai prediksi. Nilai MAE yang semakin kecil menunjukkan bahwa rata-rata kesalahan prediksi model semakin rendah.

**b. Root Mean Squared Error (RMSE)**

RMSE merupakan akar kuadrat dari rata-rata kuadrat selisih antara nilai aktual dan prediksi. Metrik ini memberikan penalti lebih besar terhadap kesalahan prediksi yang memiliki selisih tinggi. Semakin kecil nilai RMSE, semakin rendah tingkat kesalahan model.

**c. R-squared (R²)**

R² merupakan metrik yang digunakan untuk mengukur seberapa baik model menjelaskan variasi nilai target berdasarkan variabel masukan. Nilai R² yang semakin mendekati 1 menunjukkan kemampuan model yang semakin baik dalam menjelaskan variasi target pada data evaluasi.

Ketiga metrik tersebut digunakan karena dapat memberikan gambaran performa model dari sisi rata-rata kesalahan, besarnya kesalahan kuadrat, dan kemampuan model dalam menjelaskan variasi data.

## 4. Dataset

Dataset yang digunakan dalam proyek ini adalah `Salary_Data.csv`. Dataset tersebut digunakan untuk mempelajari hubungan antara lama pengalaman kerja dengan nilai gaji.

Dataset memiliki dua kolom utama yang digunakan dalam proses pemodelan, yaitu:

| Nama Kolom | Keterangan | Jenis Variabel |
|---|---|---|
| YearsExperience | Lama pengalaman kerja dalam tahun | Fitur (X) |
| Salary | Nilai gaji | Target (y) |

Variabel `YearsExperience` digunakan sebagai masukan untuk model, sedangkan `Salary` menjadi nilai yang ingin diprediksi.

**Catatan:** File CSV harus tersedia secara terpisah dan diletakkan dalam direktori yang sama dengan notebook. Jumlah baris, jumlah data, dan satuan mata uang tidak ditetapkan dalam dokumentasi ini karena file CSV sumber tidak disertakan dalam notebook yang diperiksa.

## 5. Tools dan Library

Proyek ini dikembangkan menggunakan bahasa pemrograman Python dengan bantuan beberapa library untuk pengolahan data, pembangunan model, evaluasi, dan visualisasi.

| Tools/Library | Fungsi |
|---|---|
| Python | Bahasa pemrograman utama |
| Jupyter Notebook | Lingkungan untuk menjalankan kode |
| NumPy | Pengolahan data numerik dan array |
| Pandas | Membaca serta mengolah dataset |
| Matplotlib | Membuat visualisasi data |
| Scikit-learn | Pembagian data, standardisasi, pemodelan SVR, dan evaluasi |

Instalasi library dapat dilakukan melalui perintah berikut:

```bash
pip install numpy pandas matplotlib scikit-learn jupyter
```

## 6. Tahapan Implementasi

Proses implementasi dilakukan secara bertahap sesuai urutan kode pada notebook `SVMRegresi.ipynb`.

### 6.1 Import Library

Tahap pertama adalah mengimpor library yang dibutuhkan selama proses pemodelan.

Library NumPy digunakan untuk operasi numerik, Pandas digunakan untuk membaca dan mengelola dataset, sedangkan Matplotlib digunakan untuk menampilkan visualisasi.

Selain itu, beberapa modul dari Scikit-learn digunakan untuk membagi dataset, melakukan standardisasi, membangun model SVR, dan menghitung metrik evaluasi.

### 6.2 Membaca Dataset

Dataset dibaca menggunakan fungsi `read_csv()` dari library Pandas.

```python
data = pd.read_csv("Salary_Data.csv")
data.head()
```

Perintah tersebut digunakan untuk membaca file CSV dan menampilkan beberapa baris pertama dataset agar struktur data dapat diperiksa.

File `Salary_Data.csv` harus berada dalam direktori yang sama dengan notebook agar dapat dibaca tanpa mengubah lokasi file.

### 6.3 Menentukan Variabel X dan Y

Setelah dataset berhasil dibaca, langkah berikutnya adalah menentukan variabel independen dan dependen.

```python
X = data[["YearsExperience"]]
y = data["Salary"]
```

Variabel X berisi lama pengalaman kerja, sedangkan variabel y berisi nilai gaji yang menjadi target prediksi.

Pemisahan ini bertujuan agar model dapat mempelajari hubungan antara pengalaman kerja dan gaji.

### 6.4 Membagi Data Training dan Testing

Dataset kemudian dibagi menjadi data latih dan data uji menggunakan fungsi `train_test_split()`.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

Pembagian data menggunakan proporsi 80% untuk data latih dan 20% untuk data uji.

Data latih digunakan untuk membangun model, sedangkan data uji digunakan untuk mengevaluasi kemampuan model dalam memprediksi data yang tidak digunakan selama proses pelatihan.

Parameter `random_state=42` digunakan agar pembagian data dapat direproduksi dengan hasil yang konsisten ketika kode dijalankan kembali menggunakan dataset yang sama.

### 6.5 Standardisasi Data

Tahap selanjutnya adalah melakukan standardisasi pada fitur X dan target y menggunakan `StandardScaler`.

Standardisasi fitur:

```python
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
```

Standardisasi target:

```python
scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(
    y_train.to_numpy().reshape(-1, 1)
).ravel()
```

Pada tahap ini, `fit_transform()` digunakan untuk mempelajari parameter standardisasi sekaligus mengubah data latih. Sementara itu, `transform()` digunakan pada data uji menggunakan parameter yang telah diperoleh dari data latih.

Standardisasi target dilakukan agar skala nilai gaji tidak terlalu memengaruhi proses pembelajaran model SVR.

### 6.6 Membangun dan Melatih Model SVR

Setelah proses standardisasi selesai, model SVR dibuat menggunakan kernel RBF dengan parameter yang telah ditentukan.

```python
model_svr = SVR(
    kernel="rbf",
    C=100,
    epsilon=0.1,
    gamma="scale"
)

model_svr.fit(X_train_scaled, y_train_scaled)
```

Parameter yang digunakan memiliki fungsi masing-masing. Kernel RBF digunakan untuk mempelajari pola nonlinier, C mengatur penalti kesalahan, epsilon menentukan batas toleransi kesalahan, dan gamma mengatur pengaruh setiap data latih dalam pembentukan fungsi regresi.

Selanjutnya, fungsi `fit()` digunakan untuk melatih model menggunakan data pengalaman kerja dan gaji yang telah distandardisasi.

### 6.7 Melakukan Prediksi dan Evaluasi Model

Setelah model selesai dilatih, tahap berikutnya adalah melakukan prediksi terhadap data uji.

```python
y_pred_scaled = model_svr.predict(X_test_scaled).reshape(-1, 1)
y_pred = scaler_y.inverse_transform(y_pred_scaled).ravel()
```

Hasil prediksi awal masih berada dalam skala standardisasi. Oleh karena itu, dilakukan proses inverse transform agar nilai prediksi kembali ke skala gaji aslinya.

Setelah itu, dilakukan evaluasi menggunakan MAE, RMSE, dan R².

```python
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
```

Hasil prediksi kemudian disusun ke dalam DataFrame agar nilai pengalaman kerja, gaji aktual, dan prediksi SVR dapat dibandingkan secara langsung.

### 6.8 Visualisasi Hasil Prediksi

Visualisasi dilakukan menggunakan Matplotlib untuk memperlihatkan perbandingan antara data latih, data uji, dan kurva hasil prediksi model.

Pada grafik:

- Titik berwarna biru menunjukkan data latih.
- Titik berwarna oranye menunjukkan data uji.
- Garis berwarna merah menunjukkan kurva hasil prediksi SVR dengan kernel RBF.

Kurva dibentuk menggunakan 200 titik yang dibuat secara berurutan dari nilai pengalaman kerja minimum hingga maksimum pada dataset.

Visualisasi ini bertujuan untuk memberikan gambaran mengenai pola hubungan antara lama pengalaman kerja dengan nilai gaji serta bagaimana model membentuk hasil regresinya.

### 6.9 Pengujian Prediksi Data Baru

Tahap terakhir adalah menguji model menggunakan nilai pengalaman kerja baru, yaitu 5 tahun.

```python
pengalaman_baru = np.array([[5.0]])

pengalaman_baru_scaled = scaler_X.transform(pengalaman_baru)

prediksi_baru_scaled = model_svr.predict(
    pengalaman_baru_scaled
).reshape(-1, 1)

prediksi_gaji = scaler_y.inverse_transform(
    prediksi_baru_scaled
)[0, 0]
```

Nilai pengalaman kerja terlebih dahulu distandardisasi menggunakan scaler fitur yang sudah dilatih sebelumnya. Setelah itu, model menghasilkan prediksi dalam skala standardisasi yang kemudian dikembalikan ke skala gaji asli menggunakan `inverse_transform()`.

Dengan demikian, model dapat digunakan untuk memperkirakan gaji berdasarkan masukan pengalaman kerja baru.

## 7. Hasil dan Pembahasan

Berdasarkan hasil eksekusi notebook, model Support Vector Regression dengan kernel RBF menghasilkan nilai evaluasi sebagai berikut.

| Metrik Evaluasi | Hasil |
|---|---:|
| MAE | 5.994,69 |
| RMSE | 7.343,23 |
| R² | 0,8944 |

### 7.1 Analisis MAE

Nilai MAE sebesar 5.994,69 menunjukkan bahwa rata-rata selisih absolut antara hasil prediksi dengan gaji aktual pada data uji adalah sebesar 5.994,69 satuan nilai gaji.

Semakin kecil nilai MAE, semakin rendah rata-rata kesalahan prediksi yang dihasilkan oleh model.

### 7.2 Analisis RMSE

Nilai RMSE sebesar 7.343,23 menunjukkan besarnya kesalahan prediksi setelah memperhitungkan kuadrat selisih antara nilai aktual dan prediksi.

Nilai RMSE yang lebih besar daripada MAE menunjukkan bahwa terdapat kesalahan prediksi yang relatif lebih besar dibandingkan rata-rata kesalahan absolutnya.

### 7.3 Analisis R-squared

Nilai R² sebesar 0,8944 menunjukkan bahwa model mampu menjelaskan sekitar 89,44% variasi nilai gaji pada data uji berdasarkan variabel pengalaman kerja yang digunakan.

Nilai tersebut menunjukkan bahwa model memiliki kemampuan penjelasan yang cukup baik pada data pengujian. Namun, nilai R² tidak dapat diartikan sebagai persentase akurasi prediksi secara langsung.

### 7.4 Hasil Prediksi Gaji

Pengujian tambahan dilakukan dengan memasukkan pengalaman kerja selama 5 tahun.

Hasil prediksi yang diperoleh adalah:

**67.389,37 satuan nilai gaji.**

Hasil tersebut merupakan estimasi yang dihasilkan oleh model berdasarkan pola pada dataset yang digunakan. Nilainya bukan merupakan standar gaji resmi atau jaminan besaran gaji yang akan diterima seseorang di dunia kerja.

## 8. Kesimpulan

Berdasarkan implementasi yang telah dilakukan, algoritma Support Vector Regression dengan kernel Radial Basis Function dapat digunakan untuk memprediksi nilai gaji berdasarkan lama pengalaman kerja.

Proses pembangunan model dilakukan melalui beberapa tahapan, yaitu membaca dataset, menentukan fitur dan target, membagi data menjadi data latih dan data uji, melakukan standardisasi, melatih model, melakukan evaluasi, serta menampilkan visualisasi hasil prediksi.

Model yang dibangun menghasilkan nilai MAE sebesar 5.994,69, RMSE sebesar 7.343,23, dan R² sebesar 0,8944. Selain itu, pengujian terhadap pengalaman kerja selama 5 tahun menghasilkan estimasi gaji sebesar 67.389,37 satuan nilai gaji.

Secara keseluruhan, hasil tersebut menunjukkan bahwa SVR dapat mempelajari pola hubungan antara pengalaman kerja dan gaji pada dataset yang digunakan. Meskipun demikian, model ini masih memiliki keterbatasan karena hanya menggunakan satu variabel masukan, yaitu lama pengalaman kerja. Faktor lain seperti pendidikan, bidang pekerjaan, lokasi, jabatan, dan pengalaman khusus belum dimasukkan ke dalam proses pemodelan.

Pengembangan selanjutnya dapat dilakukan dengan menambahkan variabel yang lebih beragam, melakukan pencarian hyperparameter menggunakan GridSearchCV, serta membandingkan performa SVR dengan algoritma regresi lainnya agar diperoleh hasil prediksi yang lebih komprehensif.

## 9. Keterbatasan Penelitian

Beberapa keterbatasan dalam implementasi ini adalah:

1. Model hanya menggunakan satu fitur, yaitu `YearsExperience`.
2. Pembagian data menggunakan satu skenario, yaitu 80% data latih dan 20% data uji.
3. Parameter SVR ditentukan secara langsung dan belum melalui proses optimasi hyperparameter.
4. Evaluasi model hanya dilakukan menggunakan MAE, RMSE, dan R².
5. Model belum membandingkan performanya dengan algoritma regresi lain.
6. Prediksi hanya didasarkan pada pola yang tersedia dalam dataset dan belum mempertimbangkan faktor eksternal yang memengaruhi gaji.

## 10. Struktur Repository

Struktur file yang digunakan dalam proyek ini adalah:

```text
SVR-Regresi/
│
├── SVMRegresi.ipynb
├── Salary_Data.csv
└── README.md
```

Keterangan:

- `SVMRegresi.ipynb`: Notebook yang berisi seluruh proses implementasi SVR.
- `Salary_Data.csv`: Dataset yang digunakan untuk pelatihan dan pengujian model.
- `README.md`: Dokumentasi proyek yang menjelaskan teori, tahapan implementasi, hasil, dan referensi penelitian.

## 11. Referensi

Referensi berikut merupakan artikel ilmiah yang membahas penerapan Support Vector Regression, pemilihan kernel, evaluasi model, dan penerapan metode regresi. Seluruh jurnal yang dipilih memiliki peringkat akreditasi SINTA 3 atau lebih tinggi berdasarkan informasi akreditasi yang ditemukan pada portal SINTA.

[1] K. B. Adam, D. K. Silalahi, B. S. Aprillia, and H. A. Adhari, "Support Vector Regression Method for Predicting Off-Grid Photovoltaic Output Power in the Short Term," *Jurnal RESTI (Rekayasa Sistem dan Teknologi Informasi)*, vol. 6, no. 4, pp. 548–554, 2022.  
https://doi.org/10.29207/resti.v6i4.4134

[2] D. Suryani and M. Fadhilla, "Indonesian Crude Oil Price (ICP) Prediction Using Support Vector Regression Algorithm," *Jurnal RESTI (Rekayasa Sistem dan Teknologi Informasi)*, vol. 8, no. 1, pp. 127–134, 2024.  
https://doi.org/10.29207/resti.v8i1.5551

[3] A. W. Ishlah, S. Sudarno, and P. Kartikasari, "Implementasi GridSearchCV pada Support Vector Regression (SVR) untuk Peramalan Harga Saham," *Jurnal Gaussian*, vol. 12, no. 2, pp. 276–286, 2023.  
https://doi.org/10.14710/j.gauss.12.2.276-286

[4] R. A. Maulana, I. Permana, F. N. Salisah, T. K. Ahsyar, and M. Jazman, "Perbandingan Kernel Algoritma Support Vector Regression Terhadap Performa Prediksi Produksi Kelapa Sawit," *MALCOM: Indonesian Journal of Machine Learning and Computer Science*, vol. 5, no. 1, pp. 405–413, 2025.  
https://doi.org/10.57152/malcom.v5i1.1410

[5] R. A. Pangestu, A. V. Vitianingsih, S. Kacung, A. L. Maukar, and A. Noertjahyana, "Comparative Analysis of Support Vector Regression and Linear Regression Models to Predict Apple Inc. Share Prices," *Indonesian Journal of Artificial Intelligence and Data Mining*, vol. 7, no. 1, pp. 148–156, 2024.  
https://doi.org/10.24014/ijaidm.v7i1.28594

[6] W. Aprilya, M. S. N. Van Delsen, and M. Y. Matdoana, "Method of Application of Support Vector Regression in Predicting the Number of Visits of Foreign Tourists to the Province of Maluku," *Indonesian Journal of Artificial Intelligence and Data Mining*, vol. 6, no. 1, pp. 61–82, 2023.  
https://doi.org/10.24014/ijaidm.v6i1.19803

[7] T. A. D. Kurniawan, A. Setiawan, and F. Tita, "Perbandingan Kinerja Metode Support Vector Regression dan Metode Regresi Linier Berganda dalam Memprediksi BMI pada Dataset ASTHMA," *Jurnal Sains dan Edukasi Sains*, vol. 8, no. 2, pp. 133–142, 2025.  
https://doi.org/10.24246/juses.v8i2p133-142
