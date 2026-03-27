import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

"""
link = https://www.kaggle.com/datasets/jorgesandoval/wind-power-generation
Veri Seti Adi: Wind Power Generation Data
Problem:
    1 - Yilin günlerine ve saatlerine göre rüzgar enerjisi üretimini incelemek
    2 - 4 farkli Alman şirketinin rüzgar enerjisi üretimlerini karşilaştirmak
Değişkenler:
    Tarih, saat, ölçümün yapildiği şirket, Terawatt saat cinsinden üretilen güç
Veri Seti Hakkinda Kisa Tanim:
    Veri Seti yilin günü, saati ve rüzgar enerjisini üreten şirketin adini taşimaktadir.
    Bu bilgilerle yilin gününe ve saatine göre rüzgar enerjisi üretimi ve şirketlerin rüzgar enerjisi üretimlerinin kiyaslanmasi planlanmaktadir.
"""

# ------- 4 Farkli Şirketin Rüzgar Üretim Bilgilerinin Birleştirilmesi ve Veri Setine Genel Bakış ----------

# Şirket isimleri
company_names = ["50Hertz","Amprion","TenneTTSO","TransnetBW"]

# Tüm veriler indirildikten sonra dosya olarak alınıyor
all_data = [
    pd.read_csv("C:/Users/Asim/Desktop/50Hertz.csv"),
    pd.read_csv("C:/Users/Asim/Desktop/Amprion.csv"),
    pd.read_csv("C:/Users/Asim/Desktop/TenneTTSO.csv"),
    pd.read_csv("C:/Users/Asim/Desktop/TransnetBW.csv")
]

for index in range(len(company_names)):
    # Şirket verilerinin toplanması
    data = all_data[index]
    #şirket isimleri sütununu ekle
    data["Company"] = company_names[index]
    cols_to_convert = data.columns[1:97]
    data[cols_to_convert] = data[cols_to_convert].apply(pd.to_numeric, errors='coerce')
    data[cols_to_convert] = data[cols_to_convert].astype('float64')
    #her şirketin verilerini rastgele kaldırma
    for i in range(10):
        rand_row = np.random.randint(0,396)
        rand_col = np.random.randint(1,96)
        data.iloc[rand_row,rand_col] = None
    
    #her şirketin verilerine rastgele hatalar ekleme(ortalamadan çok farklı sayılar)
    for j in range(3):
        rand_row = np.random.randint(0,396)
        rand_col = np.random.randint(1,96)
        rand_value = np.random.randint(1000,4000)
        data.iloc[rand_row,rand_col] = rand_value
        
    # Zaman formatını ayarla
    data['Date'] = pd.to_datetime(data['Date'],format="%d/%m/%Y")
    all_data[index] = data

# Tüm şirket bilgilerini birleştir
firstdf = pd.concat(all_data)

firstdf.head()

firstdf.info()

print(firstdf.describe())

print("----- Veri Setinin İlk Hali -----\n",firstdf)


# ------- Veri Temizleme ve Eksik Verileri Doldurma ----------

# z skor sınırı bunu geçen değerler sıradışı kabul edilir
z_threshold = 2
# Her şirketi ekle
for index in range(len(company_names)):
    # Şirket verilerinin toplanması
    data = all_data[index]
    # Hatalı veri türü olan satırların temizlenmesi
    columns_to_check = data.columns[1:97]
    data = data[
        data[columns_to_check].apply(lambda col: col.map(lambda x: isinstance(x, (float, np.float64)))).all(axis=1)
    ]
    #ortalama değer ve standart sapma değerleri 
    time_columns = data.columns[1:97]
    means = data[time_columns].mean()
    stds = data[time_columns].std()
    #eksik bilgileri fillna ile ortalama değer ile doldurma
    data.fillna(means,inplace = True)
    #z değeri yüksek çıkan elemanları değiştirme
    data = data.infer_objects()
    z_scores = (data[time_columns] - means)/stds
    #sıradışı değerler
    outliers = (np. abs (z_scores) > z_threshold)
    #sıradışı değerin bulunup düzeltilmesi
    for col in time_columns:
        data.loc[outliers[col], col] = means[col]
    all_data[index] = data

# Tüm şirket bilgilerini birleştir
finaldf = pd.concat(all_data)

print("----- Veri Setinin Temizlenmiş Hali -----\n",finaldf)

# ------- Veri Görselleştirme ----------

# --- Şirket bazlı box plot çizimi ---

# Zaman sütunları (ilk ve son sütun hariç)
time_columns = finaldf.columns[1:-1]
# Tarih ve şirket sütunu
date_col = finaldf.columns[0]
sirket_col = finaldf.columns[-1]
# Şirket isimlerini al
sirketler = finaldf[finaldf.columns[-1]].unique()

# Her şirketin güç verilerini ayrı listelere topla
guc_listeleri = []
for sirket in sirketler:
    sirket_df = finaldf[finaldf[finaldf.columns[-1]] == sirket]
    guc_degerleri = sirket_df[time_columns].values.flatten()
    guc_degerleri = guc_degerleri[~np.isnan(guc_degerleri)]
    guc_listeleri.append(guc_degerleri)

# Box plot çizimi
plt.boxplot(guc_listeleri, tick_labels=sirketler)
plt.title("Şirket Bazlı Güç Üretim Değerleri (Box Plot)")
plt.ylabel("Güç Değeri")
plt.grid(True)
plt.show()

# --- Şirket bazında toplam güç üretimi grafiği ---
toplam_gucler = finaldf.groupby(sirket_col)[time_columns].sum().sum(axis=1)

# Bar chart çizimi
plt.figure(figsize=(8, 5))
plt.bar(toplam_gucler.index, toplam_gucler.values, color='skyblue', edgecolor='black')
plt.title("Şirket Bazında Toplam Güç Üretimi")
plt.xlabel("Şirket")
plt.ylabel("Toplam Güç (milyon Terawatt/Saat)")
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# En çok güç üreten şirketi yazdır
en_cok_ureten = toplam_gucler.idxmax()
en_cok_miktar = toplam_gucler.max()
print(f"En çok güç üreten şirket: {en_cok_ureten} ({en_cok_miktar:.2f} Terawatt/Saat)")


#----Şirketlerin günün saatine göre ortalama rüzgar enerji üretimi---

for ind in range(len(company_names)):
    # Şirket verisini filtrele
    data = finaldf[finaldf[sirket_col] == company_names[ind]]

    # Ortalama günlük üretim
    mean = data[time_columns].mean()
    plt.figure(figsize=(8, 5))
    plt.bar(mean.index, mean.values, color='red', edgecolor='black')
    plt.title(company_names[ind] + " Şirketinin Günün Saatine Göre Ortalama Güç Üretimi")
    plt.xlabel("Saat Dilimi")
    plt.ylabel("Ortalama Üretilen Güç (Terawatt/Saat)")
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

#----Tüm şirketlerin ortalama günlük rüzgar enerji üretimi----
# Satır bazında toplam üretim
finaldf['GunlukToplam'] = finaldf[time_columns].sum(axis=1)

# Şirket ve gün bazında ortalama üretimi al
gunluk_uretim = finaldf.groupby([sirket_col, date_col])['GunlukToplam'].sum()

# Şirket bazında günlük ortalama üretimi al
ortalama_gunluk_sirket = gunluk_uretim.groupby(sirket_col).mean()

print("Tüm şirketlerin ortalama günlük üretimi:")
plt.figure(figsize=(8, 5))
plt.bar(ortalama_gunluk_sirket.index, ortalama_gunluk_sirket.values, color='green', edgecolor='black')
plt.title("Şirketlerin Günlük Ortalama Güç Üretimi")
plt.xlabel("Şirket")
plt.ylabel("Ortalama Günlük Üretilen Güç (Terawatt/Saat)")
plt.grid(axis='y')
plt.tight_layout()
plt.show()

#---Aylara göre şirketlerin ortalama rüzgar enerji üretimi---

# Ay sütunu ekle
finaldf['Month'] = finaldf[date_col].dt.to_period('M')

# Satır toplamı
finaldf['GunlukToplam'] = finaldf[time_columns].sum(axis=1)

# Aylık ortalama üretim (her şirket için)
aylik_ortalamalar = finaldf.groupby(['Month', sirket_col])['GunlukToplam'].mean().unstack()

# Aylık ortalama üretimi göster
print("Aylara göre şirketlerin ortalama üretimi:")
print(aylik_ortalamalar)


#---Sabah saatlerindeki ortalama rüzgar enerjisi üretimine göre akşam saatlerindeki rüzgar enerjisi arasındaki koralasyon heatmap i---

# Saat dilimlerini seç: ilk 8 saat ve son 8 saat → 4x8 = 32 sütun
sabah_sutunlari = time_columns[:32]    # 00:00 - 08:00
aksam_sutunlari = time_columns[-32:]   # 16:00 - 24:00

# Sadece bu saatlerdeki veriler
sabah_df = finaldf[sabah_sutunlari]
aksam_df = finaldf[aksam_sutunlari]

# Ortalama üretimler
sabah_ort = sabah_df.mean(axis=1)
aksam_ort = aksam_df.mean(axis=1)

# DataFrame oluştur
saat_df = pd.DataFrame({
    'SabahOrtalama': sabah_ort,
    'AksamOrtalama': aksam_ort
})

# Korelasyon matrisi ve heatmap
corr_matrix = saat_df.corr()

plt.figure(figsize=(5, 4))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", square=True)
plt.title("Sabah ve Akşam Ort. Rüzgar Enerji Korelasyonu")
plt.show()

# --- Ortalama, Medyan, Standart Sapma Değerlerinin Hesaplanması
median = finaldf[time_columns].median()
means = finaldf[time_columns].mean()
stds = finaldf[time_columns].std()
variance = finaldf[time_columns].var()
"""print("Medyan Değerleri: ",median)
print("Ortalama Değerleri: ",means)
print("Standart Sapma Değerleri: ",stds)
print("Varyans Değerleri: ",variance)"""

print("Medyan Değerleri:")
plt.figure(figsize=(8, 5))
plt.bar(median.index, median.values, color='purple', edgecolor='black')
plt.title("Saatlere Göre Güç Üretiminin Medyan Değerleri")
plt.xlabel("Saatler")
plt.ylabel("Medyan")
plt.grid(axis='y')
plt.tight_layout()
plt.show()
print("Ortalama Değerleri:")
plt.figure(figsize=(8, 5))
plt.bar(means.index, means.values, color='purple', edgecolor='black')
plt.title("Saatlere Göre Güç Üretiminin Ortalama Değerleri")
plt.xlabel("Saatler")
plt.ylabel("Ortalama")
plt.grid(axis='y')
plt.tight_layout()
plt.show()
print("Standart Sapma Değerleri:")
plt.figure(figsize=(8, 5))
plt.bar(stds.index, stds.values, color='purple', edgecolor='black')
plt.title("Saatlere Göre Güç Üretiminin Standart Sapma Değerleri")
plt.xlabel("Saatler")
plt.ylabel("Standart Sapma")
plt.grid(axis='y')
plt.tight_layout()
plt.show()
print("Varyans Değerleri:")
plt.figure(figsize=(8, 5))
plt.bar(variance.index, variance.values, color='purple', edgecolor='black')
plt.title("Saatlere Göre Güç Üretiminin Varyans Değerleri")
plt.xlabel("Saatler")
plt.ylabel("Varyans")
plt.grid(axis='y')
plt.tight_layout()
plt.show()

"""
Sonuç: Veri seti ile aya ve günün saatine göre güç üretimleri incelendi.
Şirketlerin günlük ortalama güç üretimleri karşılaştırıldı. Bu verilere göre
dört şirket arasından en çok günlük ortalama güç üretimine sahip olan şirket TenneTTSO
oldu. TenneTTSO aynı zamanda bu veri setinde en fazla toplam güç üretimine de sahipti. . Dört şirketinde bu veri seti içerisinde en fazla günlük ortalama güç üretimine sahip
olduğu ay 2020 yılının şubat ayı olduğunu görüldü. Yine bu dört şirketin asgari güç üretiminin
2019 yılının ağustos ayında olduğu da görüldü. Öğle saatlerinde (11.00-13.00) arası varyansın yüksek
olduğunu ve güç üretiminin ortalamadan uzak değerler alabileceği görüldü. 

"""