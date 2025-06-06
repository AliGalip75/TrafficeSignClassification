from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense

class TrafficSignNet:
    @staticmethod
    def build(width, height, depth, classes):

        # sıralı katmanlardan oluşan bir model
        model = Sequential()

        inputShape = (height, width, depth)
        chanDim = -1

        model.add(Conv2D(8, (3, 3), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        #model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(16, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(16, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(32, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(32, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # Flatten -> Son konvolüsyon katmanından çıkan 2D veriyi tek boyuta çevirir. Örn: (8, 8, 32) → (2048,) Böylece Dense (tam bağlantılı) katmanlarla çalışabilir.
        model.add(Flatten())
        model.add(Dense(128, kernel_regularizer=l2(0.001)))
        model.add(Activation("relu"))
        model.add(BatchNormalization())

        model.add(Dense(64, kernel_regularizer=l2(0.001)))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))

        model.add(Dense(classes))
        model.add(Activation("softmax"))
        return model