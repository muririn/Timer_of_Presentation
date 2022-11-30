
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers import Conv2D, Reshape, MaxPooling2D, Dropout

# モデル構築
model = Sequential()

model.add(Reshape((28,28,1), input_shape=(28,28)))
model.add(Conv2D(32,(3,3)))
model.add(Activation("relu"))
model.add(Conv2D(32,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.5))

model.add(Conv2D(16,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(784))
model.add(Activation("relu"))
model.add(Dropout(0.5))

model.add(Dense(10))
model.add(Activation("softmax"))

# MNISTをロード
from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = np.array(X_train)/255
X_test = np.array(X_test)/255

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

#コンパイル
model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])

#学習実行
hist = model.fit(X_train, y_train, batch_size=200, verbose=1, 
                 epochs=1, validation_split=0.1)

#評価
score = model.evaluate(X_test, y_test, verbose=1)
print("正解率(acc)：", score[1])

#モデルを保存
model.save("MNIST_1.h5")