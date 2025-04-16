
import matplotlib
matplotlib.use("Agg")
from pyimagesearch.trafficsignnet import TrafficSignNet
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
from skimage import transform
from skimage import exposure
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import os

def load_split(basePath, csvPath):
    data = []
    labels = []

    # CSV dosyasının içeriğini yükle, ilk satırı atla (başlık satırı)
    # Satırları karıştır (aksi halde aynı sınıfa ait örnekler sıralı olur)
    rows = open(csvPath).read().strip().split("\n")[1:]
    random.shuffle(rows)

    # CSV dosyasının satırları üzerinde döngü
    for (i, row) in enumerate(rows):
        try:
            # Noktalı virgül kullanarak kolonları ayır
            components = row.strip().split(";") # ['41', '39', '6', '5', '35', '34', '1', 'Train/1/00001_00067_00010.png']

            # Son iki eleman label ve imagePath
            label, imagePath = components[-2:]

            # Görüntü dosyasının tam yolunu oluştur
            imagePath = os.path.sep.join([basePath, imagePath])
            image = io.imread(imagePath)

            # Görüntüyü en-boy oranını korumadan 32x32 piksel boyutuna getir
            # ve Kontrast Kısıtlı Adaptif Histogram Eşitleme (CLAHE) uygula
            image = transform.resize(image, (32, 32))
            image = exposure.equalize_adapthist(image, clip_limit=0.1)

            data.append(image)
            labels.append(int(label))
        except Exception as e:
            print(f"[ERROR] Skipping row due to error: {e}")
            continue

    # data ve labels'ı numpyArray'e çevir
    data = np.array(data)
    labels = np.array(labels)
    if len(labels) == 0:
        print("[ERROR] No valid labels found.")
        exit()

    # data and labels'ı tuple olarak dön
    return (data, labels)


# komut satırında parametre vermek için
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, # gtsrb-germantraffic-sign
	help="path to input GTSRB")
ap.add_argument("-m", "--model", required=True, # output/trafficsignnet.model
	help="path to output model")
ap.add_argument("-p", "--plot", type=str, default="plot.png", # output/plot.png
	help="path to training history plot")
args = vars(ap.parse_args())


# Eğitim için epoch sayısı, öğrenme oranı ve batch boyutu
NUM_EPOCHS = 30
INIT_LR = 1e-3
BS = 64

# Etiket isimlerini yükle
labelNames = open("signnames.csv").read().strip().split("\n")[1:]
labelNames = [l.split(",")[1] for l in labelNames]

# **********Eğitim ve test verilerini al(konsolda dataset parametresine "gtsrb-germantraffice-sign" adresini kendinize göre verin)**********
# Benim açımdan : python train.py --dataset "C:\\Users\galip\Desktop\Traffice_Sign_Recognition\gtsrb-germantraffice-sign" --model output/model.model --plot output/plot.png
trainPath = os.path.join(args["dataset"], "Train.csv")
testPath = os.path.join(args["dataset"], "Test.csv")

#trainPath = "C:\\Users\galip\Desktop\Traffice_Sign_Recognition\gtsrb-germantraffice-sign\Train.csv"
#testPath = "C:\\Users\galip\Desktop\Traffice_Sign_Recognition\gtsrb-germantraffice-sign\Test.csv"

# Eğitim ve test verilerini yükle
print("[INFO] loading training and testing data...")
(trainX, trainY) = load_split(args["dataset"], trainPath)
(testX, testY) = load_split(args["dataset"], testPath)

# Verileri [0, 1] arasına çek
trainX = trainX.astype("float32") / 255.0
testX = testX.astype("float32") / 255.0

# Eğitim ve test etiketlerini one-hot kodlamasına dönüştür
numLabels = len(np.unique(trainY))
trainY = to_categorical(trainY, numLabels)
testY = to_categorical(testY, numLabels)

# Her sınıftaki toplam görüntü sayısını hesapla
# ve sınıf ağırlıklarını saklamak için sözlük oluştur
classTotals = trainY.sum(axis=0)
classWeight = dict()

# Tüm sınıflar üzerinde döngü yap ve sınıf ağırlıklarını hesapla
for i in range(0, len(classTotals)):
    classWeight[i] = classTotals.max() / classTotals[i]

# Veri artırma için görüntü üreticisini oluştur
aug = ImageDataGenerator(
	rotation_range=10,
	zoom_range=0.15,
	width_shift_range=0.1,
	height_shift_range=0.1,
	shear_range=0.15,
	horizontal_flip=False,
	vertical_flip=False,
	fill_mode="nearest")


print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / (NUM_EPOCHS * 0.5))
model = TrafficSignNet.build(width=32, height=32, depth=3,
	classes=numLabels)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# training
print("[INFO] training network...")
H = model.fit(
	aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY),
	steps_per_epoch=trainX.shape[0] // BS,
	epochs=NUM_EPOCHS,
	class_weight=classWeight,
	verbose=1)

# Değerlendirme
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=BS)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), target_names=labelNames))

# Modeli parametre olarak verilen yere kaydet
model.save(args["model"])

# Eğitilen modelin verilerini matplotlib ile göster
N = np.arange(0, NUM_EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["accuracy"], label="train_acc")
plt.plot(N, H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])