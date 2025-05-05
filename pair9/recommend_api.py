from fastapi import FastAPI, HTTPException
import pandas as pd
from recommend_model import train_and_save_model

app = FastAPI()

# Global değişken olarak önerileri tut
recommendations_df = None

@app.post("/train")
def train_model():
    """
    Modeli eğit ve önerileri oluştur
    """
    try:
        _, _, recommendations = train_and_save_model()
        global recommendations_df
        recommendations_df = recommendations
        return {"message": "Model trained and recommendations generated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/recommend/{customer_id}")
def get_recommendations(customer_id: str):
    """
    Belirli bir müşteri için top 5 ürün önerisi döndür
    """
    try:
        global recommendations_df
        
        # Eğer öneriler yüklü değilse, CSV'den oku
        if recommendations_df is None:
            try:
                recommendations_df = pd.read_csv('data/top5_recommendations.csv')
            except FileNotFoundError:
                raise HTTPException(
                    status_code=404, 
                    detail="Recommendations not found. Please train the model first using /train endpoint"
                )
        
        # Müşterinin önerilerini bul
        customer_recommendations = recommendations_df[
            recommendations_df['musteri_id'] == customer_id
        ].sort_values('satin_alma_olasiligi', ascending=False)
        
        if len(customer_recommendations) == 0:
            raise HTTPException(
                status_code=404,
                detail=f"No recommendations found for customer {customer_id}"
            )
        
        # Önerileri formatlayıp döndür
        recommendations = []
        for _, row in customer_recommendations.iterrows():
            recommendations.append({
                "urun_id": row['urun_id'],
                "urun_kategori_adi": row['urun_kategori_adi'],
                "satin_alma_olasiligi": float(row['satin_alma_olasiligi'])
            })
        
        return {
            "customer_id": customer_id,
            "recommendations": recommendations
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 