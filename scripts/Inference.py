# Keras
from tensorflow import keras

# Biblioteki pomocnicze
import numpy as np
import matplotlib.pyplot as plt

# Pobieranie danych testowych z bazy MNIST
mnist_digits = keras.datasets.mnist;
_, (test_images, test_labels) = mnist_digits.load_data()

# Pobieranie wybranego obrazu oraz odpowiadającej mu etykiety
# Dzielenie przez 255.0 służy przeskalowaniu obrazka do wartości 0-1
# (jak w przypadku danych treningowych)
testImageIndex = 600;
test_image = test_images[testImageIndex] / 255.0;
actual_label = test_labels[testImageIndex]

# TensorFlow wymaga serii obrazów. 
# Dlatego konwertujemy obraz do tablicy o wymiarach [1, 28, 28]
test_image_for_tensorflow = np.expand_dims(test_image, 0);

# Wczytywanie zapisanego wcześniej modelu
inputPath = 'trained_model'
model = keras.models.load_model(inputPath);

# Analiza obrazu w celu rozpoznania cyfry
prediction_result = model.predict(test_image_for_tensorflow)
predicted_label = np.argmax(prediction_result)

# Wyświetlanie wyników
plt.figure()
plt.xticks([])
plt.yticks([])
plt.grid(False)
plt.imshow(test_image, cmap=plt.cm.binary)
#plt.xlabel('Właściwa: ' + str(actual_label) + ', rozpoznana: ' + str(predicted_label), fontsize=20)

plt.title('Handwritten digit recognition', fontsize=22)
plt.xlabel('Actual: ' + str(actual_label) + ', predicted: ' + str(predicted_label), fontsize=20)

#plt.savefig('img_' + str(testImageIndex) + '.png')
plt.show()