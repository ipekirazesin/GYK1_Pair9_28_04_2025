import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import tensorflow as tf
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, roc_auc_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from eda_analysis import df
import joblib

# SHAP görselleştirmeleri için dizin oluştur
os.makedirs('shap_png', exist_ok=True)

# Veri seçimi
feature_names = ['quantity', 'unit_price', 'discount', 'net_spent',
                 'avg_discount_by_product', 'avg_net_spent_by_product']
X = df[feature_names]
y = df['is_returned']

# Eğitim-test bölünmesi
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

# Ölçekleme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Dengesizlik çözümü (SMOTE)
smote = SMOTE(random_state=42, sampling_strategy=0.3)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

# Model tanımı
model = Sequential([
    Dense(32, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    Dropout(0.4),
    Dense(16, activation='relu', kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(1, activation='sigmoid', kernel_regularizer=l2(0.01))
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC()]
)

# Callbacks
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True, min_delta=0.001, monitor='val_loss'),
    ReduceLROnPlateau(patience=3, factor=0.2, min_lr=1e-5, monitor='val_loss')
]

def train_model():
    # Model eğitimi
    model.fit(
        X_train_balanced, y_train_balanced,
        epochs=50,
        batch_size=32,
        validation_split=0.25,
        callbacks=callbacks,
        verbose=0
    )
    
    # Performans metriklerini hesapla
    y_test_pred_proba = model.predict(X_test_scaled, verbose=0)
    y_test_pred = (y_test_pred_proba > threshold).astype(int)
    
    # SHAP analizi
    explainer = shap.Explainer(model, X_train_scaled, feature_names=feature_names)
    shap_values = explainer(X_test_scaled)

    # Özellik önemlilik grafiği
    shap.plots.bar(shap_values, max_display=6, show=False)
    plt.title('Genel Özellik Önemliliği')
    plt.tight_layout()
    plt.savefig('shap_png/shap_summary.png')
    plt.close()

    # Riskli (iade olasılığı yüksek) siparişlerin SHAP açıklamaları
    risky_indices = np.where(y_test_pred == 1)[0]
    if len(risky_indices) > 0:
        for idx in risky_indices[:5]:
            try:
                shap.plots.bar(shap_values[idx], show=False)
                plt.title(f'Sipariş #{idx} için Özellik Önemliliği')
                plt.tight_layout()
                plt.savefig(f'shap_png/shap_analysis_{idx}.png')
                plt.close()
            except Exception:
                continue
    
    metrics = {
        'roc_auc': float(roc_auc_score(y_test, y_test_pred_proba)),
        'classification_report': classification_report(y_test, y_test_pred, output_dict=True)
    }
    
    return metrics

def predict(input_data):
    # Giriş verisini ölçekle
    input_scaled = scaler.transform(input_data)
    
    # Tahmin yap
    prediction_proba = model.predict(input_scaled, verbose=0)
    prediction = (prediction_proba > threshold).astype(int)
    
    return prediction[0][0], prediction_proba[0][0]

# Modeli ve scaler'ı kaydet
os.makedirs('models', exist_ok=True)
model.save('models/return_prediction_model.h5')
joblib.dump(scaler, 'models/scaler.pkl')

# Eğitim ve test verilerini kaydet
os.makedirs('data', exist_ok=True)
np.save('data/X_train_scaled.npy', X_train_scaled)
np.save('data/X_test_scaled.npy', X_test_scaled)
np.save('data/y_train.npy', y_train)
np.save('data/y_test.npy', y_test)

# 🎯 Cost-sensitive eşik
threshold = 0.3
