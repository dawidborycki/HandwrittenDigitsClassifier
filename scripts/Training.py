# Keras
from tensorflow import keras

# Biblioteki pomocnicze
import numpy as np
import matplotlib.pyplot as plt

# Pobieranie modelu
mnist_digits = keras.datasets.mnist;
(train_images, train_labels), _ = mnist_digits.load_data()

# Nazwy klas odpowiadają kolejnym cyfrom od 0 do 9
class_names = np.arange(10);

# Prezentacja danych treningowych 
rowCount = 10;
colCount = 30;

# Przygotowanie okna 
plt.figure()

# Prezentacja kolejnych obrazków z etykietami
for i in range(rowCount * colCount):
    # Wyświetlanie poszczególnych cyfr w tabeli (rowCount x colCount)
    plt.subplot(rowCount, colCount, i+1)    
   
    # Wyłączenie etykiet osi i siatki
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)

    # Wyświetlanie cyfr i etykiet
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])

# Wyświetlenie rysunku
plt.show()

# Normalizacja danych wejściowych
train_images = train_images / 255.0;

# Definiowanie modelu
model = keras.Sequential([
    # Transformacja obrazu (28, 28) do wektora o długości 784 = 28*28
    keras.layers.Flatten(input_shape=(28, 28)),

    # Ukryta warstwa obliczeniowa zawiera 64 węzłów
    # z funkcją aktywacji sigmoid
    keras.layers.Dense(64, activation='tanh'),

    # Ukryta warstwa obliczeniowa zawiera 128 węzłów
    # z funkcją aktywacji tangens hiperbpoliczny
    keras.layers.Dense(128, activation='sigmoid'),

    # Warstwa z 10 węzłami wyjściowymi (klasami)
    # z funkcją aktywacji softmax
    keras.layers.Dense(len(class_names), activation='softmax')
])

# Przygotowanie modelu
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# Trening
model.fit(train_images, train_labels, epochs=10)

# Zapisywanie modelu do pliku w formacie tf
modelOutputPath = 'trained_model';
model.save(modelOutputPath, save_format='tf')