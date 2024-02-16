import numpy as np
DEBUG = True
printd = lambda x: print(x) if DEBUG else None

class LeNet5:
    def __init__(self):

        # Input image shape
        self.input_shape = (32, 32, 1)  # RGB image
        printd("Input shape: ")
        printd(self.input_shape)

        # Convolutional layer 1 parameters
        self.conv1_filters = 6  # Number of filters
        self.conv1_kernel_size = (5, 5)  # Kernel size

        # Pooling layer 1 parameters
        self.pool1_filter_size = (2, 2)  # Filter size for max pooling

        # Convolutional layer 2 parameters
        self.conv2_filters = 16  # Number of filters
        self.conv2_kernel_size = (5, 5)  # Kernel size

        # Pooling layer 2 parameters
        self.pool2_filter_size = (2, 2)  # Filter size for max pooling

        # Fully connected layer 1 parameters
        self.fc1_neurons = 120  # Number of neurons

        # Fully connected layer 2 parameters
        self.fc2_neurons = 84  # Number of neurons

        # Output layer parameters
        self.output_neurons = 10  # Number of output neurons

        # Create the LeNet-5 model

        # Use NumPy random functions for initialization
        # Remember to choose appropriate initialization methods based on activation function
        # Convolutional layer 1 weights and biases
        self.conv1_weights = np.random.randn(self.conv1_filters, self.conv1_kernel_size[0], self.conv1_kernel_size[0], self.input_shape[2])
        self.conv1_bias = np.zeros(self.conv1_filters)
        printd("First convolutional layer weights shape: ")
        printd(self.conv1_weights.shape)

        # Convolutional layer 2 weights and biases
        self.conv2_weights = np.random.randn(self.conv2_filters, self.conv2_kernel_size[0], self.conv2_kernel_size[0], self.conv1_filters)
        self.conv2_bias = np.zeros(self.conv2_filters)
        printd("Second convolutional layer weights shape: ")
        printd(self.conv2_weights.shape)

        # Fully connected layer 1 weights and biases
        self.fc1_weights = np.random.randn(5 * 5 * self.conv2_filters, self.fc1_neurons)
        self.fc1_bias = np.zeros(self.fc1_neurons)
        printd("First fully connected layer weights shape: ")
        printd(self.fc1_weights.shape)

        # Fully connected layer 2 weights and biases
        self.fc2_weights = np.random.randn(self.fc1_neurons, self.fc2_neurons)
        self.fc2_bias = np.zeros(self.fc2_neurons)
        printd("Second fully connected layer weights shape: ")
        printd(self.fc2_weights.shape)

        # Output layer weights and biases
        self.output_weights = np.random.randn(self.fc2_neurons, self.output_neurons)
        self.output_bias = np.zeros(self.output_neurons)
        printd("Output layer weights shape: ")
        printd(self.output_weights.shape)

    def conv_layer(self, x, weights, bias, kernel_size, filters):
        output_shape = (x.shape[0], x.shape[1] - kernel_size[0] + 1, x.shape[2] - kernel_size[1] + 1, filters)
        printd("Output shape: ")
        printd(output_shape)
        y = np.zeros(output_shape)
        
        for i in range(filters):
            conv_filter = np.zeros(y.shape[1:4])
            # Apply the convolution operation
            for j in range(0, output_shape[1]):
                for k in range(0, output_shape[2]):
                    conv_filter += x[:, j:j + kernel_size[0], k:k + kernel_size[1], :] * weights[i]
                    
            # Add the bias term
            conv_filter += bias[i]
            
            # Store the result in the output array
            y[:, :, i] = conv_filter
        
        return y
    
    def max_pooling(self, x, filter_size):
        output_shape = (x.shape[0], x.shape[1] // filter_size, x.shape[2] // filter_size, x.shape[3])
        y = np.zeros(output_shape)
        
        for i in range(0, x.shape[1], filter_size):
            for j in range(0, x.shape[2], filter_size):
                for k in range(x.shape[3]):
                    y[:, i // filter_size, j // filter_size, k] = np.max(x[:, i:i + filter_size, j:j + filter_size, k])
        
        return y
    
    def forward_pass(self, x):
        printd("Forward pass")
        printd("Input shape: ")
        printd(x.shape)
        
        # Convolutional layer 1
        conv1 = self.conv_layer(x, self.conv1_weights, self.conv1_bias, self.conv1_kernel_size, self.conv1_filters)
        printd("Convolutional layer 1 output shape: ")
        printd(conv1.shape)
        
        # Activation function
        conv1 = np.maximum(conv1, 0)
        printd("Convolutional layer 1 output after ReLU: ")
        printd(conv1)
        
        # Pooling layer 1
        pool1 = self.max_pooling(conv1, self.pool1_filter_size)
        printd("Pooling layer 1 output shape: ")
        printd(pool1.shape)
        
# Test with random data
x = np.random.randn(1, 32, 32, 1)
lenet = LeNet5()
lenet.forward_pass(x)
        