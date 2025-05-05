from Database import Database
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats.mstats import winsorize


db = Database()

main_details = db.get_query1()
#print("\n--- MAIN---")
df = pd.DataFrame(main_details)
#print(df.head())
#print(df.shape)
#print(df.dtypes)
#print(df.describe())
#print(df.isnull().sum())

#transforming to datetime
df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
df["recent_order"] = pd.to_datetime(df["recent_order"], errors="coerce")
#print(df.dtypes)

#outlier analysis
def iqr_outlier_bounds(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return lower_bound, upper_bound

columns_to_check = ['avg_discount','avg_unit_price','net_spent','total_quantity','total_orders']

for col in columns_to_check:
    lower, upper = iqr_outlier_bounds(df[col])
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    #print(f"{col} outlier count: {len(outliers)}")

df_all = pd.concat([df['avg_discount'], df['avg_unit_price'], df['net_spent'], df['total_quantity'], df['total_orders']], axis=1)

sns.pairplot(df_all)
#plt.show()

for col in columns_to_check:
    plt.figure(figsize=(6, 1.5))
    sns.boxplot(data=df, x=col)
    plt.title(f"Outlier Analysis: {col}")
    #plt.show()

df_winsor = df.copy()
df_winsor['net_spent_winsorized'] = winsorize(df_winsor['net_spent'].values, limits=[0.05, 0.05])
df_winsor['net_spent_winsorized1k'] = df_winsor['net_spent']/1000
cols_to_check = ['net_spent','net_spent_winsorized','net_spent_winsorized1k']
for col in cols_to_check:
    plt.figure(figsize=(6, 1.5))
    sns.boxplot(data=df_winsor, x=col)
    plt.title(f"Aykırı Değer Analizi: {col}")
    #plt.show()

def remove_outliers(df, col):
    lower, upper = iqr_outlier_bounds(df[col])
    return df[(df[col] >= lower) & (df[col] <= upper)]

df_clean = df.copy()
for col in columns_to_check:
    df_clean = remove_outliers(df_clean, col)

df['is_outlier_spent'] = df['net_spent'].apply(
    lambda x: x < iqr_outlier_bounds(df['net_spent'])[0] or x > iqr_outlier_bounds(df['net_spent'])[1]
)

outlier_counts = df.groupby('customer_id')['is_outlier_spent'].sum()
outlier_customers = outlier_counts[outlier_counts != 0].sort_values(ascending=False)

#print(outlier_customers)

# Müşterilere ve tarihe göre sırala
df_sorted = df_winsor.sort_values(['customer_id', 'order_date'])

# Her müşteri için bir sonraki sipariş tarihini al (lead)
df_sorted['next_order_date'] = df_sorted.groupby('customer_id')['order_date'].shift(-1)

# Her siparişin 6 ay (180 gün) içinde tekrar edilip edilmediğini kontrol et
df_sorted['will_reorder_6m'] = (
    (df_sorted['next_order_date'] - df_sorted['order_date']) <= pd.Timedelta(days=180)
)

"""# Sadece son sipariş olmayanları getir (modelde kullanılabilecek satırlar)
non_last_orders = df_sorted[df_sorted['order_date'] != df['recent_order']]

# İncele
#print(non_last_orders[['customer_id', 'order_id', 'order_date', 'recent_order', 'will_reorder_6m']].head())

filtered = df_sorted[
    (df_sorted['order_date'] != df['recent_order']) &
    (df_sorted['will_reorder_6m'] == False)
]

# İncelemek için önemli sütunlar
filtered_subset = filtered[['customer_id', 'order_id', 'order_date', 'next_order_date', 'recent_order', 'will_reorder_6m']]

#print(filtered_subset.head(10))
#print(df_sorted.shape)

filtered1 = df_sorted[
    (df_sorted['order_date'] == df['recent_order']) &
    (df_sorted['will_reorder_6m'] == False)
]

# İncelemek için önemli sütunlar
filtered_subset1 = filtered1[['customer_id', 'order_id', 'order_date', 'next_order_date', 'recent_order', 'will_reorder_6m']]

#print(filtered_subset1.shape) ##88
#print(filtered_subset1.head(10))

filtered11 = df_sorted[
    (df_sorted['will_reorder_6m'] == False)
]
#print(filtered11.shape) #99 tane
"""
# 1. Verideki en büyük sipariş tarihi
max_order_date = df_sorted['order_date'].max()

# 2. Güvenle etiketlenebilecek son siparişleri işaretle
df_sorted['label_is_reliable'] = True

df_sorted.loc[
    (df_sorted['next_order_date'].isna()) &
    ((df_sorted['order_date'] + pd.DateOffset(months=6)) > max_order_date),
    'label_is_reliable'
] = False  # Etiket güvenilmez, çünkü 6 ay geçmemiş

# 3. Etiketi güvenli olanları tut
df_final = df_sorted[df_sorted['label_is_reliable']].copy()
df_final = df_final.drop(columns=['net_spent_winsorized1k', 'country', 'city'])
#print(df_final.head())
#print(df_final.shape)

filtered111 = df_final[(df_sorted['will_reorder_6m'] == False)]
#print(filtered111) #99 tane - 88 = 11 tane kalacaktı gerçek falseları koruduk

df_final = pd.get_dummies(df_final, columns=['year_quarter'], drop_first=True)
print(df_final.dtypes)

db.close()