Traffic Sign Recognition (GTSRB)
Bu proje, Almanya Trafik İşaretleri Veri Seti (GTSRB) kullanılarak trafik işaretlerini tanımayı amaçlayan bir görüntü sınıflandırma modelidir. TensorFlow kullanılarak eğitilen model, trafik işaretlerini sınıflandırmak için derin öğrenme tabanlı bir mimari kullanır.

🗂️ Veri Seti
Veri seti: GTSRB - German Traffic Sign Dataset
İndirildikten sonra şu dosyaları içerir:

Train.csv

Test.csv

Meta.csv

Train/ klasörü

Test/ klasörü

Meta/ klasörü

⚙️ Gereksinimler
Aşağıdaki Python paketlerinin kurulu olması gerekir. Tümünü tek komutla kurmak için requirements.txt dosyasını kullanabilirsiniz:

requirements.txt
makefile
Kopyala
Düzenle
python==3.9
tensorflow==2.8.0
matplotlib
scikit-learn==1.6.1
scikit-image==0.24.0
imutils==0.5.4
numpy==1.24.3
Kurulum için:

bash
Kopyala
Düzenle
pip install -r requirements.txt
🚀 Eğitim (Training)
Aşağıdaki komut ile modeli eğitebilirsiniz:

bash
Kopyala
Düzenle
python train.py --dataset "veri_seti_dizini" --model "çıktı_model_yolu" --plot "çıktı_plot_yolu"
Örnek Kullanım
bash
Kopyala
Düzenle
python train.py --dataset "C:\\Users\\galip\\Desktop\\Traffic_Sign_Recognition\\gtsrb-germantraffic-sign" --model output/model.model --plot output/plot.png
Komut Satırı Argümanları

Argüman	Açıklama
--dataset	GTSRB veri setinin yolu
--model	Eğitilen modelin kaydedileceği dosya yolu
--plot	Eğitim geçmişini gösteren grafik dosyasının yolu
