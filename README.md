# Fetal Health Prediction API 🩺

Bu proje, CTG (Kardiyotokografi) verilerini kullanarak fetüsün sağlık durumunu (Normal, Suspect, Pathological) tahmin eden bir Derin Öğrenme (PyTorch) modelidir. 

**Not:** Klinik ortamda 21 özelliğin manuel girilmesi pratik olmayacağı için, bu sistem medikal cihazlardan doğrudan JSON verisi çekip doktorun ekranına anlık risk analizi yansıtacak bir **Mikroservis (FastAPI)** mimarisinde tasarlanmıştır. Modelin eğitimi sırasında hastaların patolojik risklerini gözden kaçırmamak adına "Class Weight" (Sınıf Ağırlıklandırması) uygulanmış ve "False Negative" oranı sıfıra indirilmiştir.