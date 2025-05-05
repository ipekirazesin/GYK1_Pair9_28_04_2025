import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
import joblib
import os
from recommend_preprocessing import preprocess_data

def train_and_save_model():
    # Veriyi yükle ve preprocess
    df = preprocess_data()
    
    # Feature'ları ve target'ı ayır
    X = df.drop(columns=["satin_alindi_mi", "musteri_id", "urun_id", "urun_kategori_adi"])
    y = df["satin_alindi_mi"]
    
    # StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # SMOTE
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    # LightGBM model
    model = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        class_weight='balanced',
        random_state=42
    )
    model.fit(X_train_resampled, y_train_resampled)
    
    # Tahmin ve metrikler
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    y_pred_bin = model.predict(X_test)
    
    print("ROC AUC:", roc_auc_score(y_test, y_pred_prob))
    print("F1 Score:", f1_score(y_test, y_pred_bin))
    print(classification_report(y_test, y_pred_bin, digits=3))
    
    # Tahminleri tüm veri için yap
    df["satin_alma_olasiligi"] = model.predict_proba(scaler.transform(X))[:, 1]
    
    # Her müşteri için en yüksek 5 olasılıklı ürünü öner
    top5_df = df.sort_values(["musteri_id", "satin_alma_olasiligi"], ascending=[True, False]) \
                .groupby("musteri_id").head(5)
    
    # Precision@5 hesapla
    precision = precision_at_k(top5_df)
    print("Precision@5:", precision)
    
    # Model ve top5_df'i kaydet
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    joblib.dump(model, 'models/lightgbm_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    top5_df.to_csv('data/top5_recommendations.csv', index=False)
    
    return model, scaler, top5_df

def precision_at_k(df, k=5):
    scores = []
    for _, group in df.groupby("musteri_id"):
        precision = group["satin_alindi_mi"].head(k).sum() / k
        scores.append(precision)
    return sum(scores) / len(scores)

if __name__ == "__main__":
    model, scaler, top5_df = train_and_save_model() 