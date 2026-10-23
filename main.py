from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
import torch
import torch.nn as nn
import joblib
import numpy as np

# 1. Model Mimarisi (Ağırlıkları yükleyebilmek için sınıfı burada tekrar tanımlamalıyız)
class FetalHealthMLP(nn.Module):
    def __init__(self, input_dim=21, num_classes=3):
        super(FetalHealthMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes)
        )

    def forward(self, x):
        return self.network(x)

app = FastAPI(title="Fetal Health Prediction API")
templates = Jinja2Templates(directory="templates")

# 2. Önceden Kaydedilmiş Dosyaları RAM'e Yükleme (Uygulama başlarken 1 kez çalışır)
device = torch.device('cpu') # Canlı ortamda inference genelde CPU'da yapılır
model = FetalHealthMLP(input_dim=21, num_classes=3)
model.load_state_dict(torch.load('best_fetal_model.pth', map_location=device, weights_only=True))
model.eval() # Modeli tahmin moduna alıyoruz (Dropout ve BatchNorm davranışını değiştirir)

scaler = joblib.load('scaler.pkl')
feature_stats = joblib.load('feature_stats.pkl')
feature_names = list(feature_stats.keys())

# Sonuç etiketleri
target_map = {
    0: {"label": "Normal (Sağlıklı)", "color": "success", "desc": "Fetüsün değerleri normal aralıkta."},
    1: {"label": "Suspect (Şüpheli)", "color": "warning", "desc": "Değerlerde anormallik şüphesi var, yakın takip önerilir."},
    2: {"label": "Pathological (Patolojik)", "color": "danger", "desc": "Kritik seviye! Acil tıbbi müdahale veya detaylı inceleme gerekebilir."}
}

@app.get("/")
async def home(request: Request):
    # Parametre isimlerini açıkça belirtiyoruz
    return templates.TemplateResponse(
        request=request,
        name="index.html", 
        context={"request": request, "features": feature_stats}
    )

@app.post("/predict")
async def predict(request: Request):
    form_data = await request.form()
    
    input_values = []
    # 3. Formdan gelen verileri okuma, baskılama ve ölçeklendirme
    for col in feature_names:
        # Formdan string gelen veriyi float'a çeviriyoruz
        val = float(form_data[col])
        
        # Data Leakage'i önlemek için eğitimde bulduğumuz %99.5 üst sınırı ile baskılıyoruz (clipping)
        upper_limit = feature_stats[col]['upper_clip']
        val = min(val, upper_limit)
        input_values.append(val)
    
    # Veriyi [1, 21] boyutlu numpy dizisine çevirip Scaler'dan geçiriyoruz
    input_array = np.array(input_values).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    
    # 4. PyTorch ile Tahmin (Inference)
    input_tensor = torch.tensor(input_scaled, dtype=torch.float32)
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        prediction_idx = torch.argmax(outputs, dim=1).item()
    
    result = target_map[prediction_idx]
    confidence = round(float(probabilities[prediction_idx]) * 100, 2)
    
    
    return templates.TemplateResponse(
        request=request,
        name="index.html", 
        context={
            "request": request,
            "features": feature_stats,
            "result": result,
            "confidence": confidence,
            "submitted_data": form_data
        }
    )