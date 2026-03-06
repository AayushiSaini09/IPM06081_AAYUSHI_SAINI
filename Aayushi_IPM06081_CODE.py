import pickle

def load_cifar_batch(file_path):
    with open(file_path, 'rb') as fo:
        batch = pickle.load(fo, encoding='bytes')
    
    # Extract images and reshape to (num_samples, 32, 32, 3)
    raw_images = batch[b'data']
    x = raw_images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    y = np.array(batch[b'labels'])
    
    return x, y

# Example usage:
# x_train, y_train = load_cifar_batch('data_batch_1')
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Dense
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

# ==========================================
# 1. Load Dataset and Normalize Pixel Values
# ==========================================
print("Loading and normalizing CIFAR-10 dataset...")
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize pixel values to be between 0 and 1
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Flatten labels for sklearn compatibility
y_train_flat = y_train.flatten()
y_test_flat = y_test.flatten()

# ==========================================
# 2. Unsupervised Learning: Autoencoder for Compression
# ==========================================
print("Building and training Autoencoder...")

# Input placeholder
input_img = Input(shape=(32, 32, 3))

# Encoder (Compresses the image)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x) # Compressed representation (8x8x8 = 512 dims)

# Decoder (Reconstructs the image)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

# Compile models
autoencoder = Model(input_img, decoded)
encoder = Model(input_img, encoded) # We can use this to extract features later

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Train Autoencoder (using a small number of epochs for demonstration)
autoencoder.fit(x_train, x_train,
                epochs=5,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test),
                verbose=1)



[Image of an autoencoder neural network architecture]


# ==========================================
# 3. Supervised Model: Convolutional Neural Network (CNN) on RAW Images
# ==========================================
print("Building and training CNN on raw images...")
cnn = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train CNN
history_cnn = cnn.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test), verbose=1)

# Evaluate CNN
cnn_loss, cnn_accuracy = cnn.evaluate(x_test, y_test, verbose=0)
print(f"--> CNN Accuracy (Raw Images): {cnn_accuracy * 100:.2f}%\n")

# ==========================================
# 4. Supervised Model: SVM on Compressed Features (Using PCA)
# ==========================================
# Note: We use PCA here to easily compress features down to a 1D vector suited for SVMs.
print("Extracting features using PCA and training SVM...")

# Flatten images for PCA
x_train_flat = x_train.reshape(-1, 32 * 32 * 3)
x_test_flat = x_test.reshape(-1, 32 * 32 * 3)

# Apply PCA to compress features down to 128 components
pca = PCA(n_components=128)
x_train_pca = pca.fit_transform(x_train_flat)
x_test_pca = pca.transform(x_test_flat)

# Train SVM
# Note: SVMs scale quadratically with sample size. We use a subset of 10,000 training samples to keep runtime reasonable.
subset_size = 10000
svm_classifier = SVC(kernel='rbf', C=1.0)
svm_classifier.fit(x_train_pca[:subset_size], y_train_flat[:subset_size])

# Evaluate SVM
svm_predictions = svm_classifier.predict(x_test_pca)
svm_accuracy = accuracy_score(y_test_flat, svm_predictions)
print(f"--> SVM Accuracy (PCA Compressed Features): {svm_accuracy * 100:.2f}%\n")

# ==========================================
# 5. Visualize Reconstructed Images from Autoencoder
# ==========================================
print("Visualizing reconstructions...")
decoded_imgs = autoencoder.predict(x_test)

n = 10  # Number of images to display
plt.figure(figsize=(20, 4))
for i in range(n):
    # Display original raw images
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i])
    plt.title("Original")
    plt.axis('off')

    # Display autoencoder reconstructions
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i])
    plt.title("Reconstructed")
    plt.axis('off')

plt.tight_layout()
plt.show()
