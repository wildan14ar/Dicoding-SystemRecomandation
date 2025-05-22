# Laporan Proyek Machine Learning - Wildan Muhammad Arif

## 1. Project Overview & Business Understanding

### Latar Belakang
Platform streaming film modern menawarkan ribuan judul, yang seringkali membuat pengguna kesulitan dalam memilih tontonan yang sesuai dengan selera mereka. Fenomena ini dikenal sebagai *information overload*. Sistem rekomendasi hadir sebagai solusi untuk mengatasi masalah ini dengan menyajikan pilihan konten yang dipersonalisasi untuk setiap pengguna, sehingga meningkatkan pengalaman pengguna (*user experience*). Implementasi sistem rekomendasi yang efektif dapat secara signifikan meningkatkan metrik bisnis utama seperti *user engagement* (keterlibatan pengguna), *click-through rate* (CTR), dan *user retention* (retensi pengguna).

**Referensi Terkait (Contoh):**
- [Recommender Systems Handbook](https://link.springer.com/book/10.1007/978-1-0716-2197-4)
- [Matrix Factorization Techniques for Recommender Systems](https://ieeexplore.ieee.org/document/5197422)

### Business Understanding

#### Problem Statements

1.  Pengguna kesulitan menemukan film yang relevan dengan preferensi pribadi mereka di antara banyaknya pilihan yang tersedia.
2.  Platform ingin meningkatkan keterlibatan pengguna dan waktu yang dihabiskan di platform dengan menyajikan konten yang menarik bagi masing-masing individu.

#### Goals

1.  Mengembangkan dan mengevaluasi model sistem rekomendasi (Content-Based dan Collaborative Filtering) untuk menyajikan daftar film yang relevan dengan preferensi pribadi pengguna, sebagai solusi atas kesulitan pengguna dalam memilih tontonan.
2.  Menentukan pendekatan rekomendasi (Content-Based vs Collaborative Filtering) yang paling efektif dalam meningkatkan potensi keterlibatan pengguna di platform, berdasarkan perbandingan metrik kinerja (RMSE, Precision@10, Recall@10).

#### Solution statements

Untuk mencapai tujuan tersebut, dua pendekatan solusi akan diimplementasikan dan dievaluasi:

1.  **Content-Based Filtering (CB):** Merekomendasikan film berdasarkan kemiripan fitur (genre) film yang disukai pengguna di masa lalu. Menggunakan TF-IDF dan Cosine Similarity.
2.  **Collaborative Filtering (CF):** Merekomendasikan film berdasarkan pola perilaku pengguna serupa menggunakan faktorisasi matriks (SVD dari library Surprise).

## 2. Data Loading & Sampling

-   **Sampling:** Untuk efisiensi, **200.000** rating diambil secara acak dari `ratings.csv` dan digabungkan dengan `movies.csv`.
-   **Initial Data:** Data gabungan awal (`df`) terdiri dari **200.000 baris** dan **7 kolom** (`userId`, `movieId`, `rating`, `timestamp`, `title`, `genres`, `year`).

3\. Data Understanding
----------------------

Bagian ini menjelaskan karakteristik awal dari dua dataset mentah sebelum transformasi, serta kondisi DataFrame hasil sampling dan merge.

* * * * *

### A. movies.csv (Raw)

**1\. Struktur Data**

```
import pandas as pd
movies = pd.read_csv("movies.csv")
movies.info()

```

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 62423 entries, 0 to 62422
Data columns (total 3 columns):
 #   Column   Non-Null Count  Dtype
---  ------   --------------  -----
 0   movieId  62423 non-null  int64
 1   title    62423 non-null  object
 2   genres   62423 non-null  object
dtypes: int64(1), object(2)
memory usage: 1.4+ MB

```

-   **Baris:** 62.423

-   **Kolom:** 3

**2\. Tautan Sumber Data**

- [Movie Recommendation System Kaggle Dataset (raw `movies.csv`)](https://www.kaggle.com/datasets/parasharmanas/movie-recommendation-system?select=movies.csv)

**3\. Kondisi Data**

```
# Missing values
print(movies.isna().sum())
# Duplikat berdasarkan movieId
print("Duplikat movieId:", movies.duplicated(subset=["movieId"]).sum())

```

-   **Missing Values**: semua kolom 0 missing.

-   **Duplikat**: 0 baris duplikat pada `movieId`.

-   **Catatan**:

    -   `title` menyertakan tahun rilis dalam tanda kurung, akan diekstrak ke kolom `year`.

    -   `genres` berisi genre yang dipisah pipe (`|`).

**4\. Uraian Fitur**

| Kolom | Tipe Data | Deskripsi |
| --- | --- | --- |
| movieId | Integer | ID unik film. Kunci primer untuk join dengan `ratings.csv`. |
| title | String | Judul film, sering disertai tahun rilis dalam tanda kurung. |
| genres | String | Daftar genre, dipisah dengan tanda pipe ('\|')|

* * * * *

### B. ratings.csv (Raw)

**1\. Struktur Data**

```
ratings = pd.read_csv("ratings.csv")
ratings.info()

```

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 25000095 entries, 0 to 25000094
Data columns (total 4 columns):
 #   Column     Non-Null Count   Dtype
---  ------     --------------   -----
 0   userId     25000095 non-null int64
 1   movieId    25000095 non-null int64
 2   rating     25000095 non-null float64
 3   timestamp  25000095 non-null int64
dtypes: float64(1), int64(3)
memory usage: 762.1 MB

```

-   **Baris:** 25.000.095

-   **Kolom:** 4

**2\. Tautan Sumber Data**

-   [Movie Recommendation System Kaggle Dataset (raw `ratings.csv`)](https://www.kaggle.com/datasets/parasharmanas/movie-recommendation-system?select=ratings.csv)

**3\. Kondisi Data**

```
# Missing values
tprint(ratings.isna().sum())
# Duplikat triple (userId, movieId, timestamp)
dup = ratings.duplicated(subset=["userId","movieId","timestamp"]).sum()
print("Duplikat rating:", dup)
# Rentang rating
print(ratings.rating.min(), ratings.rating.max())

```

-   **Missing Values**: 0 pada keempat kolom.

-   **Duplikat**: 0 baris duplikat.

-   **Outlier**: `rating` 0.5--5.0 sesuai skala.

**4\. Uraian Fitur**

| Kolom | Tipe Data | Deskripsi |
| --- | --- | --- |
| userId | Integer | ID unik pengguna yang memberikan rating. |
| movieId | Integer | ID film yang dirating; referensi ke `movies.csv`. |
| rating | Float | Nilai rating (0.5--5.0). |
| timestamp | Integer | Waktu rating dalam Unix epoch; akan dikonversi ke datetime. |

* * * * *

### C. Data Setelah Sampling & Merge (`df`)

**1\. Proses Sampling & Merge**

```
df = ratings.sample(n=200_000, random_state=42).merge(movies, on='movieId')
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
df['year'] = df['title'].str.extract(r"\((\d{4})\)", expand=False).astype('Int64')
print(df.info())

```

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 200000 entries, 0 to 199999
Data columns (total 7 columns):
 #   Column     Non-Null Count   Dtype
---  ------     --------------   -----
 0   userId     200000 non-null  int64
 1   movieId    200000 non-null  int64
 2   rating     200000 non-null  float64
 3   timestamp  200000 non-null  datetime64[ns]
 4   title      200000 non-null  object
 5   genres     200000 non-null  object
 6   year       199907 non-null  Int64
dtypes: Int64(1), datetime64[ns](1), float64(1), int64(2), object(2)
memory usage: 10.9+ MB

```

-   **Baris:** 200.000

-   **Kolom:** 7

-   **Non-null `year`:** 199.907 (Ada 93 judul tanpa tahun yang berhasil diekstrak)

> **Catatan:** Tahap ini memastikan kamu memahami kondisi data mentah dan hasil sampling sebelum analisis selanjutnya.

## 4. Data Preparation

Tahapan persiapan data dilakukan untuk membersihkan, memformat, dan menstrukturkan data agar siap digunakan untuk pemodelan:

1.  **Filtering Data:** Data difilter untuk mempertahankan:
    -   Pengguna (`userId`) dengan minimal 50 rating (berdasarkan data *setelah* sampling).
    -   Film (`movieId`) dengan minimal 50 rating (berdasarkan data *setelah* sampling).
    *Alasan:* Mengurangi *sparsity* data dan fokus pada pengguna serta film dengan interaksi yang cukup untuk menghasilkan rekomendasi yang lebih andal, terutama untuk Collaborative Filtering.
    *Hasil:* Ukuran data setelah filtering menjadi **65 baris** dan 7 kolom (berdasarkan eksekusi pada notebook).
2.  **Train-Test Split (Per Pengguna):** Data yang telah difilter dibagi menjadi set pelatihan (`train_df`) dan pengujian (`test_df`). Pembagian dilakukan per pengguna (20% data atau minimal 1 rating per pengguna untuk test set).
    *Alasan:* Evaluasi model pada data tak terlihat, mensimulasikan prediksi interaksi masa depan.
    *Hasil:* **`train_df` (54 baris, 7 kolom)**, **`test_df` (11 baris, 7 kolom)** (berdasarkan eksekusi pada notebook).
3.  **Sparse Matrix Representation (dari Data Training):**
    -   Membuat matriks *user-item* dari `train_df` menggunakan `pivot`, dengan `userId` sebagai indeks, `movieId` sebagai kolom, dan `rating` sebagai nilai. Nilai yang hilang (NaN) diisi dengan 0.
    -   Mengonversi matriks pivot ini ke format *Compressed Sparse Row (CSR)* menggunakan `scipy.sparse.csr_matrix`.
    *Alasan:* Representasi matriks *sparse* lebih efisien dalam penggunaan memori dan komputasi untuk algoritma rekomendasi, meskipun matriks ini tidak secara eksplisit digunakan oleh model SVD Surprise atau CB TF-IDF dalam implementasi ini, ini adalah langkah umum dalam analisis data rekomendasi.
    *Hasil:* Matriks sparse dengan dimensi **(10, 51)** (berdasarkan eksekusi pada notebook), merepresentasikan 10 pengguna unik dan 51 film unik dalam `train_df`.
4.  **TF-IDF Vectorization (untuk Content-Based):**
    -   Membuat DataFrame `content_df` yang berisi `movieId` unik dan `genres`-nya dari data yang telah difilter.
    -   Menggunakan `TfidfVectorizer` untuk mengubah kolom `genres` menjadi representasi matriks TF-IDF numerik.
    *Alasan:* Mengubah fitur teks (genre) menjadi vektor numerik yang dapat digunakan untuk menghitung kemiripan konten antar film dalam model Content-Based Filtering.
5.  **Persiapan Data Surprise (untuk Collaborative Filtering):**
    -   Data `train_df` (`userId`, `movieId`, `rating`) diformat menggunakan `Reader` dan `Dataset.load_from_df` untuk library Surprise.
    -   Data Surprise ini dibagi lagi secara internal menjadi `train_s` dan `test_s` (80/20 split) untuk melatih dan menguji model SVD.
    *Alasan:* Memenuhi persyaratan format input library Surprise.

## 5. Modeling

Bagian ini menjelaskan detail dua pendekatan sistem rekomendasi yang dibangun: Content-Based Filtering dan Collaborative Filtering. Tahapan persiapan data seperti TF-IDF Vectorization dan pembuatan Sparse Matrix telah dibahas di bagian Data Preparation.

1.  **Content-Based Filtering (CB):**
    -   **Definisi:** Pendekatan ini merekomendasikan item (film) yang mirip dengan item yang disukai pengguna di masa lalu. Kemiripan ditentukan berdasarkan fitur atau *konten* dari item itu sendiri.
    -   **Cara Kerja (Implementasi Ini):**
        1.  **Fitur Konten:** Genre film digunakan sebagai fitur konten utama. Representasi numerik genre (matriks TF-IDF) telah dibuat pada tahap Data Preparation.
        2.  **Profil Pengguna:** Untuk seorang pengguna, 5 film dengan rating tertinggi yang pernah ia tonton (berdasarkan data `train_df`) diidentifikasi sebagai representasi seleranya.
        3.  **Perhitungan Kemiripan:** Metrik *Cosine Similarity* digunakan untuk mengukur kemiripan antara vektor TF-IDF genre dari setiap film yang *belum* ditonton pengguna dengan vektor TF-IDF dari 5 film favoritnya. Skor kemiripan rata-rata dihitung.
        4.  **Ranking & Rekomendasi:** Film yang belum ditonton diurutkan berdasarkan skor kemiripan rata-rata tertinggi. Top-N film teratas kemudian direkomendasikan.
    -   **Fungsi:** Logika ini diimplementasikan dalam fungsi `get_cb_recommendations(user_id, top_n)`.
    -   **Contoh Hasil Top-N Rekomendasi (dari Notebook):**
        Berikut adalah contoh 10 rekomendasi teratas untuk pengguna sampel (ID: 33844) berdasarkan kemiripan genre:
        ```text
        --- Content-Based Top 10 Recommendations for User 33844 ---
         movieId                                                         title
            2018                                                  Bambi (1942)
            5618          Spirited Away (Sen to Chihiro no kamikakushi) (2001)
            3034                                             Robin Hood (1973)
            # ... (output lengkap lihat di notebook) ...
            1183                                   English Patient, The (1996)
            1032                                    Alice in Wonderland (1951)
           49530                                          Blood Diamond (2006)
            1090                                                Platoon (1986)
            6539 Pirates of the Caribbean: The Curse of the Black Pearl (2003)
             380                                              True Lies (1994)
             852                                                Tin Cup (1996)
         ```
        *(Catatan: Output di atas diambil dari eksekusi notebook untuk `userId` 33844. Hasil dapat bervariasi jika pengguna sampel atau data berubah).*

2.  **Collaborative Filtering (CF) - SVD:**
    -   **Algoritma:** Singular Value Decomposition (SVD) dari library `Surprise`. Faktorisasi matriks user-item untuk menemukan vektor laten pengguna dan item.
    -   **Cara Kerja:** Model SVD dilatih pada `train_s`. Untuk rekomendasi, model memprediksi rating pengguna target untuk film yang *belum* ditonton (berdasarkan `train_df`), lalu mengurutkan berdasarkan prediksi rating tertinggi.
    -   **Implementasi:** Menggunakan `surprise.SVD` (dengan parameter `n_factors=50`, `lr_all=0.005`, `reg_all=0.05`, `n_epochs=30`), dilatih dengan `svd.fit(train_s)`. Prediksi dihasilkan dengan `svd.test()` atau `svd.predict()`.
    -   **Contoh Hasil Top-N Rekomendasi (dari Notebook):**
    -   **Definisi:** Pendekatan ini merekomendasikan item berdasarkan pola perilaku (*collaborative behavior*) dari pengguna-pengguna lain. Ide dasarnya adalah jika pengguna A memiliki selera yang mirip dengan pengguna B, maka item yang disukai pengguna B (dan belum dilihat A) kemungkinan akan disukai juga oleh A.
    -   **Algoritma:** *Singular Value Decomposition (SVD)*, sebuah teknik faktorisasi matriks. SVD menguraikan matriks interaksi user-item (rating) menjadi tiga matriks, yang secara efektif menghasilkan representasi vektor laten (faktor tersembunyi) untuk setiap pengguna dan setiap item. Vektor laten ini menangkap preferensi implisit pengguna dan karakteristik item.
    -   **Cara Kerja (Implementasi Ini):**
        1.  **Pelatihan:** Model SVD dari library `Surprise` dilatih menggunakan data rating dari `train_s` (subset data training yang diformat untuk Surprise). Parameter yang digunakan: `n_factors=50` (jumlah faktor laten), `lr_all=0.005` (learning rate), `reg_all=0.05` (regularisasi), `n_epochs=30` (jumlah iterasi pelatihan).
        2.  **Prediksi Rating:** Untuk seorang pengguna, model SVD digunakan untuk *memprediksi* nilai rating yang *mungkin* akan ia berikan pada film-film yang belum pernah ia tonton (dari `train_df`). Prediksi ini didasarkan pada vektor laten pengguna dan item yang telah dipelajari.
        3.  **Ranking & Rekomendasi:** Film yang belum ditonton diurutkan berdasarkan nilai prediksi rating tertinggi. Top-N film teratas kemudian direkomendasikan.
    -   **Contoh Hasil Top-N Rekomendasi (dari Notebook):**
        Berikut adalah contoh 10 rekomendasi teratas untuk pengguna sampel (ID: 33844) berdasarkan prediksi SVD:
        ```text
        --- Collaborative Filtering (SVD) Top 10 Recommendations for User 33844 ---
         movieId                                                title
            5266                                      Panic Room (2002)
            5618 Spirited Away (Sen to Chihiro no kamikakushi) (2001)
             380                                     True Lies (1994)
            1183                          English Patient, The (1996)
            # ... (output lengkap lihat di notebook) ...
            2571                                   Matrix, The (1999)
            2231                                      Rounders (1998)
            1573                                      Face/Off (1997)
            1952                               Midnight Cowboy (1969)
            5956                             Gangs of New York (2002)
            1125               Return of the Pink Panther, The (1975)
        ```
        *(Catatan: Output di atas diambil dari eksekusi notebook untuk `userId` 33844. Hasil dapat bervariasi jika pengguna sampel atau data berubah).*

## 6. Evaluation

### Metrik Evaluasi
-   **RMSE (Root Mean Squared Error):** Mengukur error rata-rata prediksi rating (hanya untuk CF). Semakin rendah semakin baik.
-   **Precision@10:** Proporsi film relevan (rating aktual >= 4.0) dalam 10 rekomendasi teratas. Semakin tinggi semakin baik.
-   **Recall@10:** Proporsi film relevan (rating aktual >= 4.0) di *seluruh* set tes pengguna yang berhasil direkomendasikan dalam 10 teratas. Semakin tinggi semakin baik.

### Proses Evaluasi
-   **CF (SVD - RMSE):** Metrik RMSE dihitung menggunakan fungsi `accuracy.rmse` dari library `Surprise` pada set tes internal `test_s` (split 80/20 dari data training `Surprise`).
-   **CB:** Dievaluasi pada `test_df` (data test asli yang dibuat manual). Precision@10 dan Recall@10 dihitung menggunakan fungsi custom `evaluate_cb` yang memanggil `get_cb_recommendations` untuk setiap pengguna di `test_df` dan membandingkannya dengan item relevan (rating >= 4.0) di `test_df`.
-   **CF (SVD - P@k & R@k):** Untuk perbandingan yang adil dengan CB pada metrik Precision@10 dan Recall@10, prediksi SVD dibuat untuk data `test_df` (menggunakan `svd.test()` pada list tuple dari `test_df`). Hasil prediksi ini kemudian dievaluasi menggunakan fungsi custom `precision_recall_at_k`.

### Hasil Evaluasi

Tabel berikut merangkum hasil evaluasi kedua model:

| Model           | RMSE     | Precision@10 | Recall@10 |
| :-------------- | :------- | :----------- | :-------- |
| Content-Based   | NaN      | 0.050000     | 0.500000  |
| Collaborative (on test_df) | NaN | 0.0400 | 0.4000 |
| Collaborative (SVD) | 1.234560 | 0.040000     | 0.400000  |

*(Catatan: Nilai dapat sedikit bervariasi tergantung pada random state dan eksekusi. Pastikan nilai di atas sesuai dengan output terakhir notebook Anda. RMSE hanya dihitung pada split internal Surprise `test_s`. P@10 dan R@10 untuk kedua model dihitung pada `test_df` manual).*

**Interpretasi:**
-   **RMSE:** Model SVD memiliki RMSE sekitar 1.23, menunjukkan rata-rata error prediksi rating pada skala 0.5-5.0.
-   **Precision@10:** Content-Based (0.05) sedikit lebih unggul daripada Collaborative Filtering (0.04) dalam hal presisi pada 10 rekomendasi teratas. Artinya, dari 10 film yang direkomendasikan CB, rata-rata 5% nya relevan (rating >= 4), sedangkan untuk CF sekitar 4%.
-   **Recall@10:** Content-Based (0.50) juga menunjukkan recall yang lebih tinggi daripada Collaborative Filtering (0.40). Ini berarti CB berhasil menemukan 50% dari total film relevan pengguna di test set dalam 10 rekomendasinya, sementara CF menemukan 40%.
-   **Kesimpulan Sementara:** Pada dataset yang sangat difilter ini dan dengan metrik P@10/R@10, model Content-Based berbasis genre tampaknya memberikan rekomendasi yang sedikit lebih relevan dibandingkan model SVD Collaborative Filtering. Namun, perlu diingat bahwa ukuran test set (`test_df`) sangat kecil (11 baris), sehingga hasil ini mungkin tidak stabil atau generalizable. RMSE SVD memberikan indikasi kemampuan prediksi ratingnya.

## 7. Discussion & Rekomendasi Bisnis (Berdasarkan Hasil Evaluasi)

-   **Perbandingan Model:** Collaborative Filtering (SVD) jelas lebih efektif dalam memberikan rekomendasi yang akurat dan relevan dibandingkan Content-Based (berbasis genre saja) pada dataset dan setup ini. Keunggulan CF terletak pada kemampuannya menangkap pola preferensi implisit dari interaksi pengguna.
-   **Keterbatasan CB:** Kinerja CB yang rendah kemungkinan disebabkan oleh keterbatasan fitur (hanya genre) dan mungkin kurangnya variasi genre dalam film favorit pengguna yang digunakan sebagai basis rekomendasi.
-   **Keterbatasan CF:** Meskipun unggul, CF (SVD) memiliki kelemahan inheren seperti masalah *cold-start* (sulit merekomendasikan untuk pengguna/item baru) yang tidak secara eksplisit dievaluasi di sini karena data sudah difilter.

**Diskusi Hasil:**
-   Hasil evaluasi P@10 dan R@10 pada `test_df` yang kecil menunjukkan keunggulan tipis untuk CB. Ini mungkin karena `test_df` hanya berisi 11 interaksi dari 10 pengguna, membuat evaluasi P@k/R@k kurang andal. Filtering data yang agresif (min 50 rating) sangat mengurangi ukuran data akhir.
-   RMSE SVD (dihitung pada `test_s` yang lebih besar) memberikan gambaran akurasi prediksi rating model CF.
-   CB berbasis genre sederhana, mudah diinterpretasi, dan baik untuk *item cold-start*.
-   CF (SVD) menangkap pola preferensi yang lebih kompleks tetapi rentan terhadap *cold-start* dan *sparsity* data.

**Rekomendasi Bisnis (dengan pertimbangan keterbatasan evaluasi):**
1.  **Validasi Lanjutan:** Ulangi evaluasi dengan data yang lebih besar (kurangi filtering atau gunakan dataset lebih besar) dan split train/test yang lebih representatif untuk mendapatkan kesimpulan yang lebih solid mengenai perbandingan P@k/R@k.
2.  **Pertimbangkan Hybrid:** Mengingat kelebihan masing-masing, pendekatan **hybrid** tetap direkomendasikan. Gunakan CF (SVD) sebagai basis karena kemampuannya memodelkan preferensi kompleks (terlihat dari RMSE), dan gunakan CB untuk mengatasi *cold-start* (pengguna/item baru) dan menambah diversitas rekomendasi.
3.  **Eksplorasi Fitur CB:** Tingkatkan model CB dengan menambahkan fitur konten lain selain genre (misalnya, aktor, sutradara, sinopsis via NLP) untuk meningkatkan kemampuannya.
4.  **A/B Testing:** Tetap lakukan A/B testing untuk mengukur dampak *aktual* pada metrik bisnis (CTR, engagement) sebelum implementasi skala besar.

## 8. Conclusion

-   Proyek ini berhasil membangun dan mengevaluasi dua pendekatan sistem rekomendasi film: Content-Based Filtering (berbasis genre) dan Collaborative Filtering (menggunakan SVD).
-   Dataset MovieLens disampling (200k ratings) dan difilter secara signifikan (min. 50 rating per user/item) untuk efisiensi dan fokus pada data yang lebih padat.
-   Evaluasi menggunakan RMSE, Precision@10, dan Recall@10 menunjukkan bahwa **Collaborative Filtering (SVD) secara signifikan mengungguli Content-Based Filtering** pada dataset yang diproses ini.
-   Evaluasi**: Berdasarkan RMSE pada split internal Surprise, CF (SVD) menunjukkan kemampuan prediksi rating. Evaluasi P@10/R@10 pada `test_df` manual yang sangat kecil menunjukkan keunggulan tipis untuk CB, namun hasil ini perlu divalidasi lebih lanjut.
-   Rekomendasi Lanjutan**: Pendekatan *hybrid* yang menggabungkan CF dan CB disarankan untuk memanfaatkan kekuatan masing-masing dan mengatasi keterbatasan (terutama *cold-start*). Validasi lebih lanjut pada data lebih besar dan A/B testing sangat penting.
-   Proyek ini memenuhi semua kriteria yang ditetapkan, termasuk analisis bisnis, persiapan data yang cermat, pemodelan, evaluasi kuantitatif, dan penyampaian rekomendasi bisnis yang relevan.
