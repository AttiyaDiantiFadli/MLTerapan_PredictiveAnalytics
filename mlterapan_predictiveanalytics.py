# -*- coding: utf-8 -*-
"""MLTerapan_PredictiveAnalytics.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1EHdZIFiBxy2gx7fJ8OA9KkwlE2shsu-t

Proyek Pertama - Predictive Analytics
Attiya Dianti Fadli MC189D5X0806

Prediksi Penyakit Jantung
#### Sumber Data : [kaggle](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)

## 1. Import Library yang Dibutuhkan
"""

!pip install -q kaggle

#Import Load data Library
from google.colab import files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile

# Import train test split
from sklearn.model_selection import train_test_split

# Import Minmaxscaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder

#Import Model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score

"""## 2. Data Understanding

### 2.1 Data Loading
"""

files.upload()

!mkdir ~/.kaggle
!cp kaggle.json ~/.kaggle
!chmod 600 ~/.kaggle/kaggle.json

!kaggle datasets download -d fedesoriano/heart-failure-prediction

zip_ref = zipfile.ZipFile('/content/heart-failure-prediction.zip', 'r')
zip_ref.extractall('/content/')
zip_ref.close()

df = pd.read_csv('/content/heart.csv')

"""### 2.2 Exploratory Data Analysis (EDA)

#### 2.2.1 EDA - Deskripsi Variabel
"""

df

"""Dari dataframe di atas, kita dapat melihat bahwa dataset ini memiliki 12 kolom. Di antaranya:

- `Age` : Usia pasien (dalam tahun)
- `Sex` : Jenis kelamin pasien (`M` = Male, `F` = Female)
- `ChestPainType` : Tipe nyeri dada yang dialami (`ATA`, `NAP`, `ASY`, dll)
- `RestingBP` : Tekanan darah saat istirahat (dalam mm Hg)
- `Cholesterol` : Kadar kolesterol serum (dalam mg/dl)
- `FastingBS` : Gula darah puasa (`1` jika >120 mg/dl, selain itu `0`)
- `RestingECG` : Hasil elektrokardiogram saat istirahat (`Normal`, `ST`, `LVH`)
- `MaxHR` : Detak jantung maksimum yang dicapai
- `ExerciseAngina` : Apakah pasien mengalami angina saat olahraga (`Y` atau `N`)
- `Oldpeak` : Depresi ST yang diinduksi oleh olahraga relatif terhadap istirahat
- `ST_Slope` : Kemiringan segmen ST selama latihan (`Up`, `Flat`, `Down`)
- `HeartDisease` : Target variabel (`1` jika pasien memiliki penyakit jantung, `0` jika tidak)
"""

df.info()

"""Dataset ini terdiri dari **918 baris** dan **12 kolom**, dengan rincian sebagai berikut:

- Total kolom bertipe `int64` : 6 kolom  
- Total kolom bertipe `object` (kategori/teks) : 5 kolom  
- Total kolom bertipe `float64` : 1 kolom  
- Tidak terdapat data yang hilang (semua kolom memiliki 918 nilai non-null)
"""

df.describe()

"""Fungsi `describe()` memberikan informasi statistik ringkasan pada kolom numerik dalam dataset. Berikut adalah penjelasan dari hasil statistik yang ditampilkan:

- `Count` menunjukkan bahwa seluruh kolom memiliki 918 entri, yang berarti tidak ada data yang hilang pada kolom-kolom tersebut.
- `Mean` atau rata-rata usia pasien adalah sekitar 53 tahun, dengan tekanan darah istirahat (RestingBP) rata-rata 132 mmHg, dan kadar kolesterol rata-rata sekitar 198 mg/dl.
- `FastingBS` (status gula darah puasa) memiliki rata-rata 0.23, menunjukkan bahwa sebagian besar pasien memiliki kadar gula puasa normal (karena nilainya dominan 0).
- `MaxHR` atau detak jantung maksimum yang dicapai memiliki rata-rata 136 denyut per menit.
- `Oldpeak`, yang menggambarkan depresi ST, memiliki rata-rata sekitar 0.89, menunjukkan tingkat stres jantung yang relatif rendah secara umum.
- `HeartDisease`, sebagai target variabel, memiliki rata-rata sekitar 0.55, yang mengindikasikan distribusi data pasien dengan dan tanpa penyakit jantung cukup seimbang.

Nilai-nilai `min`, `25%`, `50%`, `75%`, dan `max` menunjukkan rentang serta persebaran data:
- `Min` menunjukkan nilai terendah pada tiap kolom.
- `25%` adalah kuartil pertama, yaitu 25% dari data berada di bawah nilai ini.
- `50%` adalah median atau nilai tengah dari data.
- `75%` adalah kuartil ketiga, yaitu 75% dari data berada di bawah nilai ini.
- `Max` menunjukkan nilai tertinggi pada tiap kolom.
"""

df.shape

"""| Jumlah Baris | Jumlah Kolom |
| ------ | ------ |
| 918 | 12 |

#### 2.2.2 EDA - Menangani Missing Value dan Outliers
"""

df.duplicated().sum()

"""dapat dilihat bahwa terdapat 0 data yang terduplikat."""

df.HeartDisease.value_counts(normalize=True)

"""Itu berarti sekitar 55% pasien memiliki penyakit jantung (`HeartDisease` = 1) dan 45% tidak memilikinya (`HeartDisease` = 0)."""

df.isnull().sum()

"""fungsi `df.isnull().sum()` menunjukkan bahwa **tidak ada kolom yang memiliki nilai kosong**, sehingga **tidak diperlukan proses imputasi atau penghapusan data** pada tahap ini.

**Visualisasi Outlier**
"""

df_outlier=df.select_dtypes(exclude=['object'])
for column in df_outlier:
        plt.figure()
        sns.boxplot(data=df_outlier, x=column)

"""Visualisasi boxplot dilakukan pada fitur numerik untuk mendeteksi adanya outlier.  
Beberapa fitur menunjukkan keberadaan outlier yang dapat dipertimbangkan untuk ditangani pada tahap selanjutnya.
"""

Q1 = df.select_dtypes(include=['number']).quantile(0.25)
Q3 = df.select_dtypes(include=['number']).quantile(0.75)
IQR=Q3-Q1
df = df[~((df.select_dtypes(include=['number'])<(Q1-1.5*IQR))|(df.select_dtypes(include=['number'])>(Q3+1.5*IQR))).any(axis=1)]

"""Dilakukan penghapusan outlier menggunakan metode **Interquartile Range (IQR)** pada fitur numerik.  
Baris dengan nilai di luar rentang Q1 - 1.5×IQR dan Q3 + 1.5×IQR dihapus untuk meningkatkan kualitas data.
"""

df.shape

"""Jumlah Datasets setalah menghapus Outlier: `588, 12`

#### 2.2.3 EDA - Univariate Analysis
"""

heart_disease_counts = df.HeartDisease.value_counts(normalize=True)

# Membuat grafik batang
heart_disease_counts.plot(kind='bar')
plt.title('Persentase Penyakit Jantung')
plt.xlabel('Penyakit Jantung (0 = Tidak, 1 = Ya)')
plt.ylabel('Persentase')
plt.xticks(rotation=0)
plt.show()

"""Visualisasi ini menunjukkan distribusi kelas pada fitur `HeartDisease`.  
Grafik batang digunakan untuk melihat proporsi pasien dengan dan tanpa penyakit jantung (Penyakit Jantung `(0 = Tidak, 1 = Ya)`).
"""

df.select_dtypes(include=['number']).mean().plot(kind='bar', figsize=(10, 5))
plt.show()

"""Visualisasi ini menunjukkan rata-rata dari setiap fitur numerik dalam dataset. Tujuannya adalah untuk mendapatkan gambaran awal tentang skala dan sebaran nilai dari fitur numerik.

#### 2.2.4 EDA - Multivariate Analysis
"""

sns.pairplot(df, kind='reg', diag_kind='kde')

"""Dilakukan visualisasi hubungan antar fitur menggunakan `pairplot` untuk melihat pola hubungan linier serta distribusi data masing-masing fitur."""

fig, ax = plt.subplots(figsize=(10, 8))
corr_matrix = df.select_dtypes(include=['number']).corr().round(2)
sns.heatmap(corr_matrix, ax=ax, annot=True, cmap='RdBu_r', linewidths=0.5)
ax.set_title("Matriks Korelasi untuk Fitur Numerik", fontsize=20)
plt.show()

"""Visualisasi korelasi antar fitur numerik dilakukan untuk mengidentifikasi hubungan antar variabel. Hasilnya membantu dalam memahami fitur yang memiliki keterkaitan kuat, baik positif maupun negatif, terhadap target maupun fitur lainnya.

## 3. Data Preparation

### 3.1 Data Clening
"""

# Mengonversi label HeartDisease menjadi boolean numerik: 1 jika ada penyakit, 0 jika tidak
df['HeartDisease'] = df['HeartDisease'].apply(lambda x: 1 if x == 1 else 0)

"""Pada tahap ini, kolom `HeartDisease` diubah menjadi nilai numerik boolean, di mana:
- Nilai 1 menunjukkan adanya penyakit jantung
- Nilai 0 menunjukkan tidak ada penyakit jantung
Peringatan `SettingWithCopyWarning` muncul karena perubahan dilakukan langsung pada DataFrame, namun ini tidak mempengaruhi hasil akhir.

"""

fitur = df.loc[:, df.columns != 'HeartDisease']
target = df['HeartDisease']

print(fitur.shape, target.shape)

"""Pada tahap ini, dataset dipisahkan menjadi fitur (X) dan target (y).
- **Fitur** berisi semua kolom kecuali 'HeartDisease'
- **Target** adalah kolom 'HeartDisease'

Dimensi fitur: `(588, 11)`, dimensi target: `(588,)`

### 3.2 Train-Test-Split
"""

X_train, X_test, y_train, y_test = train_test_split(
    fitur, target,
    test_size=0.2,
    random_state=42,
    stratify=target
)

total_data = len(fitur)
print(f"Jumlah total dataset: {total_data}")
print(f"Jumlah data latih: {X_train.shape[0]}")
print(f"Jumlah data uji: {X_test.shape[0]}")

"""Data dibagi menjadi data latih (80%) dan data uji (20%) menggunakan `train_test_split`.  
- Jumlah total dataset: 588
- Jumlah data latih: 470
- Jumlah data uji: 118

Pemisahan dilakukan dengan stratifikasi berdasarkan target untuk memastikan distribusi yang seimbang.

### 3.3 Encoding dan Normalisasi
"""

kolom_numerik = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']
kolom_kategorikal = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']

"""Pada tahap ini, kolom-kolom dalam dataset dikelompokkan menjadi dua kategori:
- **Kolom Numerik**: `Age`, `RestingBP`, `Cholesterol`, `FastingBS`, '`MaxHR`, `Oldpeak`
- **Kolom Kategorikal**: `Sex`, `ChestPainType`, `RestingECG`, `ExerciseAngina`, `ST_Slope`

"""

encoder = OneHotEncoder(drop='first', handle_unknown='ignore')
X_train_cat = encoder.fit_transform(X_train[kolom_kategorikal])
X_test_cat = encoder.transform(X_test[kolom_kategorikal])

"""Pada tahap ini, dilakukan One Hot Encoding pada fitur kategorikal untuk mengubah variabel kategori menjadi numerik.
Fitur pertama pada setiap kategori dihapus untuk menghindari multikolinearitas, dan pengaturan `handle_unknown='ignore'` digunakan untuk menangani kategori yang tidak muncul di data latih.

"""

scaler = MinMaxScaler()
X_train_num = scaler.fit_transform(X_train[kolom_numerik])
X_test_num = scaler.transform(X_test[kolom_numerik])

"""Pada langkah ini, dilakukan normalisasi fitur numerik menggunakan `MinMaxScaler` untuk mengubah rentang nilai fitur agar berada di antara 0 dan 1.
Proses ini dilakukan pada data latih (`X_train`) dan data uji (`X_test`).

"""

X_train_final = np.hstack((X_train_num, X_train_cat.toarray()))
X_test_final = np.hstack((X_test_num, X_test_cat.toarray()))

"""Fitur numerik dan kategorikal digabungkan menggunakan `np.hstack()` untuk membentuk dataset final yang akan digunakan dalam pelatihan model.

## 4. Model Development
"""

!pip install lazypredict

from lazypredict.Supervised import LazyClassifier

klasifikasi_otomatis = LazyClassifier(verbose=0, ignore_warnings=True)
hasil_model, hasil_prediksi = klasifikasi_otomatis.fit(X_train, X_test, y_train, y_test)
print(hasil_model.sort_values("Accuracy", ascending=False))

"""Pada tahap ini, dilakukan pelatihan berbagai model menggunakan `LazyClassifier` dari library `lazypredict`. Model-model ini dilatih secara otomatis dengan data latih dan diuji menggunakan data uji.

**Hasil Evaluasi:**
- Model dengan akurasi tertinggi adalah **PassiveAggressiveClassifier** dengan akurasi 0.87.
- Model lain seperti **BernoulliNB** dan **ExtraTreesClassifier** memiliki akurasi yang cukup tinggi (0.86).
- Model dengan akurasi terendah adalah **DummyClassifier** dengan akurasi 0.58.

Hasil ini memberikan gambaran model mana yang paling efektif untuk digunakan dalam proyek ini.

"""

plt.figure(figsize=(12, 8))
sorted_results = hasil_model.sort_values(by='Accuracy', ascending=True)

# Buat barplot
sns.barplot(
    x=sorted_results['Accuracy'],
    y=sorted_results.index,
    palette='viridis'
)

plt.title('Perbandingan Akurasi Model (LazyClassifier)', fontsize=14)
plt.xlabel('Akurasi')
plt.ylabel('Model')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

"""- Model **PassiveAggressiveClassifier** menunjukkan akurasi terbaik, sedangkan **DummyClassifier** berada di urutan terakhir dengan akurasi terendah.

"""

models = pd.DataFrame(index=['accuracy_score'],
                      columns=['PassiveAggressiveClassifier', 'BernoulliNB', 'ExtraTreesClassifier', 'RandomForestClassifier', 'LogisticRegression'])

"""Pada tahap ini, dibuat sebuah `DataFrame` dengan `pandas` untuk menyimpan hasil evaluasi (akurasi) dari berbagai model yang telah dilatih sebelumnya. DataFrame ini memiliki baris `accuracy_score` dan kolom yang mewakili setiap model yang diuji, seperti:
- **PassiveAggressiveClassifier**
- **BernoulliNB**
- **ExtraTreesClassifier**
- **RandomForestClassifier**
- **LogisticRegression**

DataFrame ini akan digunakan untuk menyimpan skor akurasi model-model yang telah dilatih dan dievaluasi.

### 4.1 Passive Aggressive Classifier
"""

model_pa = PassiveAggressiveClassifier(max_iter=1000, random_state=42)
model_pa.fit(X_train_final, y_train)

"""Pada tahap ini, model **PassiveAggressiveClassifier** dilatih dengan data latih (`X_train_final` dan `y_train`)."""

pa_pred = model_pa.predict(X_test_final)
models.loc['accuracy_score', 'PassiveAggressiveClassifier'] = accuracy_score(y_test, pa_pred)

"""Pada tahap ini, dilakukan prediksi menggunakan model **PassiveAggressiveClassifier** yang telah dilatih sebelumnya. Prediksi dilakukan pada data uji (`X_test_final`), dan hasil prediksi dihitung akurasinya menggunakan `accuracy_score`.

### 4.2 Bernoulli NB
"""

model_nb = BernoulliNB()
model_nb.fit(X_train_final, y_train)

"""model **BernoulliNB** dilatih dengan data latih (`X_train_final` dan `y_train`)."""

nb_pred = model_nb.predict(X_test_final)
models.loc['accuracy_score', 'BernoulliNB'] = accuracy_score(y_test, nb_pred)

"""dilakukan prediksi menggunakan model **BernoulliNB** yang telah dilatih sebelumnya. Prediksi dilakukan pada data uji (`X_test_final`), dan hasil prediksi dihitung akurasinya menggunakan `accuracy_score`.

### 4.3 ExtraTrees Classifier
"""

model_et = ExtraTreesClassifier(n_estimators=100, random_state=42)
model_et.fit(X_train_final, y_train)

"""model **ExtraTreesClassifier** dilatih dengan data latih (`X_train_final` dan `y_train`)."""

et_pred = model_et.predict(X_test_final)
models.loc['accuracy_score', 'ExtraTreesClassifier'] = accuracy_score(y_test, et_pred)

"""dilakukan prediksi menggunakan model **ExtraTressClassifier** yang telah dilatih sebelumnya. Prediksi dilakukan pada data uji (`X_test_final`), dan hasil prediksi dihitung akurasinya menggunakan `accuracy_score`.

### 4.4 Random Forest Classifier
"""

model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
model_rf.fit(X_train_final, y_train)

"""model **RandomForestClassifier** dilatih dengan data latih (`X_train_final` dan `y_train`)."""

rf_pred = model_rf.predict(X_test_final)
models.loc['accuracy_score', 'RandomForestClassifier'] = accuracy_score(y_test, rf_pred)

"""dilakukan prediksi menggunakan model **randomForestClassifier** yang telah dilatih sebelumnya. Prediksi dilakukan pada data uji (`X_test_final`), dan hasil prediksi dihitung akurasinya menggunakan `accuracy_score`.

### 4.5 Logistic Regression
"""

model_lr = LogisticRegression(max_iter=1000, random_state=42)
model_lr.fit(X_train_final, y_train)

"""model **LogisticRegression** dilatih dengan data latih (`X_train_final` dan `y_train`)."""

lr_pred = model_lr.predict(X_test_final)
models.loc['accuracy_score', 'LogisticRegression'] = accuracy_score(y_test, lr_pred)

"""dilakukan prediksi menggunakan model **LogisticRegression** yang telah dilatih sebelumnya. Prediksi dilakukan pada data uji (`X_test_final`), dan hasil prediksi dihitung akurasinya menggunakan `accuracy_score`.

## 5. Evaluasi Model
"""

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Dictionary untuk model dan prediksi
model_dict = {
    'PassiveAggressive': model_pa,
    'BernoulliNB': model_nb,
    'ExtraTrees': model_et,
    'RandomForest': model_rf,
    'LogisticRegression': model_lr
}

# Loop evaluasi
for name, model in model_dict.items():
    y_pred = model.predict(X_test_final)

    print(f"===== Evaluasi Model: {name} =====")
    print(f"Accuracy       : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision      : {precision_score(y_test, y_pred):.4f}")
    print(f"Recall         : {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score       : {f1_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

"""### 📌 Kesimpulan Evaluasi Model

Berdasarkan hasil evaluasi terhadap lima model terpilih menggunakan metrik **Accuracy**, **Precision**, **Recall**, dan **F1-score**, dapat disimpulkan sebagai berikut:

1. **Logistic Regression**
   - Merupakan model dengan performa terbaik.
   - Memiliki **accuracy tertinggi** sebesar **86%**.
   - Menunjukkan keseimbangan yang sangat baik antara **precision** (80.36%) dan **recall** (90.00%).
   - Cocok digunakan jika ingin meminimalkan false negative (misalnya untuk kasus diagnosis penyakit jantung).

2. **Random Forest**
   - Memberikan hasil akurasi cukup tinggi yaitu **83%**.
   - Precision dan recall juga tinggi dan seimbang.
   - Stabil, mampu menangani data kompleks dan tidak mudah overfitting.

3. **Bernoulli Naive Bayes**
   - Akurasi mencapai **83%**, dengan **recall tertinggi kedua** (88.00%).
   - Cocok digunakan ketika lebih penting untuk mendeteksi semua kasus positif meskipun dengan precision yang sedikit lebih rendah (75.86%).

4. **Extra Trees Classifier**
   - Akurasi sebesar **82%**.
   - F1-score cukup kuat (**80.37%**) dengan recall juga tinggi (86.00%).
   - Alternatif ringan dibanding Random Forest dengan performa mendekati.

5. **Passive Aggressive Classifier**
   - Memiliki akurasi paling rendah di antara lima model yaitu **78%**.
   - Recall juga lebih rendah (**70.00%**) yang berarti masih banyak kasus positif tidak terdeteksi.
   - Kurang direkomendasikan sebagai model utama.

"""

plt.figure(figsize=(10, 6))
models.loc['accuracy_score'].plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Perbandingan Accuracy Model', fontsize=16)
plt.xlabel('Model', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

"""dilakukan visualisasi perbandingan akurasi antar model menggunakan bar chart.  
Grafik ini menunjukkan model mana yang memiliki akurasi tertinggi berdasarkan hasil evaluasi sebelumnya.

- Sumbu x mewakili berbagai model yang diuji.
- Sumbu y menunjukkan nilai akurasi masing-masing model.

"""