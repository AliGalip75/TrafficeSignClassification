# Traffic Sign Recognition (GTSRB)

Bu proje, **Almanya Trafik İşaretleri Veri Seti (GTSRB)** kullanılarak trafik işaretlerini tanımayı amaçlayan bir görüntü sınıflandırma modelidir. TensorFlow kullanılarak eğitilen model, trafik işaretlerini sınıflandırmak için derin öğrenme tabanlı bir mimari kullanır.

---

## 🗂️ Veri Seti

Veri seti: [GTSRB - German Traffic Sign Dataset](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)  
İndirildikten sonra şu dosyaları içerir (Bu dosyaları tek bir dizin halinde ana klasöre ekleyin):

- `Train.csv`
- `Test.csv`
- `Meta.csv`
- `Train/` klasörü
- `Test/` klasörü
- `Meta/` klasörü

---

## ⚙️ Gereksinimler

Aşağıdaki Python paketlerinin kurulu olması gerekir (Python 3.9 ile sorunsuz çalışır). Tümünü tek komutla kurmak için `requirements.txt` dosyasını kullanabilirsiniz:

**Kurulum için**:
```
pip install -r requirements.txt
```

## 🚀 Eğitim

Aşağıdaki komut ile modeli eğitebilirsiniz:

python train.py --dataset "veri_seti_dizini" --model "çıktı_model_yolu" --plot "çıktı_plot_yolu"

### Örnek Kullanım

python train.py --dataset "C:\\Users\\galip\\Desktop\\Traffic_Sign_Recognition\\gtsrb-germantraffic-sign" --model output/model.model --plot output/plot.png

#### Argüman	Açıklama

```
--dataset	GTSRB veri setinin yolu
--model	Eğitilen modelin kaydedileceği dosya yolu
--plot	Eğitim geçmişini gösteren grafik dosyasının yolu
```

## 🔍 Tahmin 

Eğitilmiş modeli kullanarak yeni bir trafik işareti görselinin sınıfını tahmin edebilirsiniz. Bunun için predict.py dosyasındaki image_path değişkenine kendi görselinizin yolunu girmeniz yeterlidir:

# predict.py
```
image_path = "your_image.png"
sign_name, confidence = predict_traffic_sign(image_path)
```
Sonuç olarak konsolda şu şekilde bir çıktı alırsınız:
```
Tahmin Edilen İşaret: Speed limit (30km/h)
Güven Skoru: 98.7%
```
Not: predict_traffic_sign() fonksiyonunun model ve sınıf isimlerini düzgün şekilde yüklediğinden emin olun.  
