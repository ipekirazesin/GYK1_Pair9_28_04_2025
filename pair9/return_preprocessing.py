import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
import os
from Database import Database

db = Database()

main_details = db.get_query2()
#print("\n--- MAIN---")
df = pd.DataFrame(main_details)

"""print(df.head())
print(df.shape)
print(df.isnull().sum())
print(df.dtypes)
print(df.describe())"""

# Outlier Analysis
numeric_cols = ['net_spent', 'quantity', 'discount', 'unit_price', 'avg_discount_by_product', 'avg_net_spent_by_product']

print("\nAykırı Değer Analizi:")
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    print(f"\n{col} - Aykırı Değer Sayısı: {len(outliers)}")
    print(f"{col} - Aykırı Değer Yüzdesi: {len(outliers)/len(df)*100:.2f}%")
    print(f"{col} - Alt Sınır: {lower:.2f}, Üst Sınır: {upper:.2f}")

    plt.figure(figsize=(10, 4))
    sns.boxplot(x=df[col])
    plt.title(f"{col} Boxplot")
    plt.show()

# İade durumuna göre analiz
print("\nİade durumuna göre sipariş sayısı:")
print(df['is_returned'].value_counts())

# İade durumuna göre görselleştirme
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='is_returned')
plt.title("İade Durumuna Göre Sipariş Sayısı")
plt.xlabel("İade Durumu (1: İade Riski Yüksek)")
plt.ylabel("Sipariş Sayısı")
plt.show()

# İndirim ve net harcama ilişkisi
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='discount', y='net_spent', hue='is_returned')
plt.title("İndirim ve Net Harcama İlişkisi")
plt.xlabel("İndirim Oranı")
plt.ylabel("Net Harcama")
plt.show()

# Ürün bazında analiz
print("\nEn çok satılan 10 ürün:")
print(df['product_name'].value_counts().head(10))

# Ürün bazında görselleştirme
plt.figure(figsize=(12, 6))
top_products = df['product_name'].value_counts().head(10)
sns.barplot(x=top_products.values, y=top_products.index)
plt.title("En Çok Satılan 10 Ürün")
plt.xlabel("Sipariş Sayısı")
plt.show()

db.close()