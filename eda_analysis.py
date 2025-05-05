import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
import os

# Grafikler için klasör oluşturma
if not os.path.exists('eda_png'):
    os.makedirs('eda_png')

# Veritabanı bağlantı bilgileri
DB_CONFIG = {
    'dialect': 'postgresql',  # veya 'mysql', 'sqlite', 'mssql' vb.
    'username': 'postgres',
    'password': 'asd123*',
    'host': 'localhost',
    'port': '5432',  # PostgreSQL için varsayılan port
    'database': 'postgres'
}

# SQLAlchemy connection string oluşturma
def get_connection_string(config):
    if config['dialect'] == 'sqlite':
        return f"sqlite:///{config['database']}.db"
    else:
        return f"{config['dialect']}://{config['username']}:{config['password']}@{config['host']}:{config['port']}/{config['database']}"

# SQLAlchemy ile veritabanına bağlanma
connection_string = get_connection_string(DB_CONFIG)
engine = create_engine(connection_string)

# SQL sorgusunu çalıştırma
query = """
WITH product_stats AS (
    SELECT 
        od.product_id,
        AVG(od.discount) AS avg_discount_by_product,
        AVG(od.unit_price * od.quantity * (1 - od.discount)) AS avg_net_spent_by_product
    FROM order_details od
    GROUP BY od.product_id
)

SELECT 
    o.order_id,
    o.customer_id,
    od.product_id,
    p.product_name,
    od.quantity,
    od.unit_price,
    od.discount,
    (od.unit_price * od.quantity * (1 - od.discount)) AS net_spent,
    ps.avg_discount_by_product,
    ps.avg_net_spent_by_product,
    CASE 
        WHEN od.discount >= 0.2 AND (od.unit_price * od.quantity * (1 - od.discount)) < 200 
        THEN 1 
        ELSE 0 
    END AS is_returned
FROM orders o
JOIN order_details od ON o.order_id = od.order_id
JOIN products p ON od.product_id = p.product_id
JOIN product_stats ps ON od.product_id = ps.product_id
WHERE od.quantity > 0
ORDER BY is_returned DESC;
"""

# Veriyi DataFrame'e yükleme
df = pd.read_sql_query(query, engine)

# Temel istatistikler
print("DataFrame'in ilk 5 satırı:")
print(df.head())
print("\nDataFrame boyutu:", df.shape)
print("\nEksik değerlerin sayısı:")
print(df.isnull().sum())
print("\nSütun veri tipleri:")
print(df.dtypes)
print("\nTemel istatistikler:")
print(df.describe())

# Aykırı değer analizi
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
    plt.savefig(f'eda_png/{col}_boxplot.png')
    plt.close()

# İade durumuna göre analiz
print("\nİade durumuna göre sipariş sayısı:")
print(df['is_returned'].value_counts())

# İade durumuna göre görselleştirme
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='is_returned')
plt.title("İade Durumuna Göre Sipariş Sayısı")
plt.xlabel("İade Durumu (1: İade Riski Yüksek)")
plt.ylabel("Sipariş Sayısı")
plt.savefig('eda_png/return_status_count.png')
plt.close()

# İndirim ve net harcama ilişkisi
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='discount', y='net_spent', hue='is_returned')
plt.title("İndirim ve Net Harcama İlişkisi")
plt.xlabel("İndirim Oranı")
plt.ylabel("Net Harcama")
plt.savefig('eda_png/discount_net_spent_scatter.png')
plt.close()

# Ürün bazında analiz
print("\nEn çok satılan 10 ürün:")
print(df['product_name'].value_counts().head(10))

# Ürün bazında görselleştirme
plt.figure(figsize=(12, 6))
top_products = df['product_name'].value_counts().head(10)
sns.barplot(x=top_products.values, y=top_products.index)
plt.title("En Çok Satılan 10 Ürün")
plt.xlabel("Sipariş Sayısı")
plt.savefig('eda_png/top_products_bar.png')
plt.close()

# Bağlantıyı kapatma
engine.dispose() 