import matplotlib
matplotlib.use("Agg")
from skimage import transform
from skimage import exposure
from skimage import io
import numpy as np
from tensorflow.keras.models import load_model

# Eğittiğimiz modeli kullanıyoruz
model = load_model('output/model.model')

def predict_traffic_sign(image_path):
    # Görüntüyü yükle ve işle
    image = io.imread(image_path)
    image = transform.resize(image, (32, 32))
    image = exposure.equalize_adapthist(image, clip_limit=0.1)
    image = image.astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)

    # Tahmin yap
    preds = model.predict(image)
    predicted_class = np.argmax(preds, axis=1)[0]

    # Etiket ismini al
    labelNames = open("signnames.csv").read().strip().split("\n")[1:]
    labelNames = [l.split(",")[1] for l in labelNames]

    return labelNames[predicted_class], preds[0][predicted_class] * 100

#***** kendi görsel yolunuz *****
image_path = "gtsrb-germantraffice-sign/Test/01490.png"
sign_name, confidence = predict_traffic_sign(image_path)
print(f"Tahmin: {sign_name}, Güven: {confidence:.2f}%")