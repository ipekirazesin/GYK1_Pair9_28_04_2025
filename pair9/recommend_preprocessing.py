import pandas as pd
import numpy as np
from Database import Database

def preprocess_data():
    # Veritabanından query3 verilerini çek
    db = Database()
    data = db.get_query3()
    db.close()
    
    # DataFrame'e çevir
    columns = ['musteri_id', 'urun_id', 'urun_kategori_id', 'urun_kategori_adi', 
              'urun_fiyati', 'urun_durumu', 'toplam_siparis', 'toplam_harcama', 
              'ortalama_harcama', 'son_siparis_tarihi', 'bu_kategorideki_harcama',
              'bu_kategorideki_miktar', 'kategori_son_siparis', 'kategori_harcama_orani',
              'urun_populerlik_siparis', 'urun_populerlik_musteri', 
              'fiyat_orani_musteriye_gore', 'satin_alindi_mi']
    
    df = pd.DataFrame(data, columns=columns)
    
    # Tarih sütunlarını datetime'a çevir
    df["son_siparis_tarihi"] = pd.to_datetime(df["son_siparis_tarihi"])
    df["kategori_son_siparis"] = pd.to_datetime(df["kategori_son_siparis"])
    
    # Recency hesapla
    bugun = df["son_siparis_tarihi"].max()
    df["recency_gun"] = (bugun - df["son_siparis_tarihi"]).dt.days
    df["kategori_recency_gun"] = (bugun - df["kategori_son_siparis"]).dt.days
    
    # NaN olan satırları temizle
    df = df[~df["toplam_siparis"].isna()].copy()
    
    # Eksik değerleri doldur
    df["kategori_recency_gun"].fillna(df["kategori_recency_gun"].max(), inplace=True)
    
    # Gereksiz sütunları sil
    df.drop(columns=["kategori_son_siparis", "son_siparis_tarihi"], inplace=True)
    
    return df 