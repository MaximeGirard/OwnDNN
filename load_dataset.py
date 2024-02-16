# Import necessary libraries
import tensorflow as tf
import matplotlib.pyplot as plt

# Download and load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Image to grayscale
x_train = tf.image.rgb_to_grayscale(x_train)
x_test = tf.image.rgb_to_grayscale(x_test)

print(x_test.shape)

# Preprocess the data (optional)
# Normalize pixel values to be between 0 and 1
x_train, x_test = x_train / 255, x_test / 255

# Convert class labels to one-hot encoded vectors
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

# Print the shapes of the train and test sets
print("X_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", x_test.shape)
print("y_test shape:", y_test.shape)

# Display the first 16 images from the training set
fig, axes = plt.subplots(4, 4, figsize=(10, 10))
for i in range(16):
    axes[i//4][i%4].imshow(x_train[i], cmap='gray')
    axes[i//4][i%4].axis('off')

plt.show()