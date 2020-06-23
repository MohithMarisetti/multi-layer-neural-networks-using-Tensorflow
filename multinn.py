# Author: Mohith Marisetti

# using tensorflow_version 2.x
import tensorflow as tf
import numpy as np

# import sys
# np.set_printoptions(threshold=sys.maxsize)

class MultiNN(object):
    def __init__(self, input_dimension):
        """
        Initialize multi-layer neural network
        :param input_dimension: The number of dimensions for each the input data sample
        """
        self.input_dimension = input_dimension
        self.weights = []
        self.biases = []
        self.activations = []
        self.loss = None

    def add_layer(self, num_nodes, activation_function):
        """
         This function adds a dense layer to the neural network
         :param num_nodes: number of nodes in the layer
         :param activation_function: Activation function for the layer
         :return: None
        """

        """
        # Understanding what add_layer() means:
        1. In Theory: We create a new layer and then add that layer to the end of our neural network.
        2. In programming: Every layer is represented only by a Weights matrix and a bias vector.
        There will be a list of many Weights matrix and bias vectors. Hence professor initialized
         the weights and biases with a empty list i.e., []. We can now simply append the new weights
         and biases to the this list implying that we created in a new layer.
        """

        # Appending the activation function for the current layer
        self.activations.append(activation_function)

        # Appending the weights for the current layer
        weights_length = len(self.weights)   # This will be based on the number of layers
        if(weights_length>0):   # Means there are weights in the last layer. So the last layer outputs will be the input to the new layer
            previous_dimension_size = self.weights[weights_length-1].shape[1]
        else:  # Means there are no weights created yet. So the previous layer will have inputs
            previous_dimension_size = self.input_dimension
        weights_in_current_layer = tf.Variable(tf.random.normal(shape=(previous_dimension_size, num_nodes)))
        self.weights.append(weights_in_current_layer)

        # Appending the bias vector for the current layer
        bias_in_current_layer = tf.Variable(tf.random.normal(shape=(1, num_nodes)))
        self.biases.append(bias_in_current_layer)






    def get_weights_without_biases(self, layer_number):
        """
        This function should return the weight matrix (without biases) for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
         :param layer_number: Layer number starting from layer 0.
         :return: Weight matrix for the given layer (not including the biases). Note that the shape of the weight matrix should be
          [input_dimensions][number of nodes]
        """
        return self.weights[layer_number]



    def get_biases(self, layer_number):
        """
        This function should return the biases for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
         :param layer_number: Layer number starting from layer 0
         :return: Weight matrix for the given layer (not including the biases). Note that the biases shape should be [1][number_of_nodes]
        """
        return self.biases[layer_number]

    def set_weights_without_biases(self, weights, layer_number):
        """
        This function sets the weight matrix for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
         :param weights: weight matrix (without biases). Note that the shape of the weight matrix should be
          [input_dimensions][number of nodes]
         :param layer_number: Layer number starting from layer 0
         :return: none
         """
        self.weights[layer_number] = weights
        return

    def set_biases(self, biases, layer_number):
        """
        This function sets the biases for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
        :param biases: biases. Note that the biases shape should be [1][number_of_nodes]
        :param layer_number: Layer number starting from layer 0
        :return: none
        """
        self.biases[layer_number] = biases
        return

    def set_loss_function(self, loss_fn):
        """
        This function sets the loss function.
        :param loss_fn: Loss function
        :return: none
        """
        self.loss = loss_fn

    def sigmoid(self, x):

        return tf.nn.sigmoid(x)

    def linear(self, x):
        return x

    def relu(self, x):
        out = tf.nn.relu(x)
        return out

    def cross_entropy_loss(self, y, y_hat):
        """
        This function calculates the cross entropy loss
        :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
         the desired (true) class.
        :param y_hat: Array of actual outputs values [n_samples][number_of_classes].
        :return: loss
        """
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=y_hat))

    def predict(self, X):
        """
        Given array of inputs, this function calculates the output of the multi-layer network.
        :param X: Array of input [n_samples,input_dimensions].
        :return: Array of outputs [n_samples,number_of_classes ]. This array is a numerical array.
        """

        """
        # WHAT predict() method does?
        Answer: self.weights will return the list of weight matrices in all the layers. We iterate
        over each of this weight matrix and multiply it will our inputs from previous layer and
        add with biases. In simple words,we calculate (X*W + b) in each layer and apply the
        activation function on it.
        """
        for layer_number in range(len(self.weights)):
            # For the first layer we calculate X*W + b
            if(layer_number == 0):
                net = tf.matmul(X, self.get_weights_without_biases(layer_number)) + self.get_biases(layer_number)
            else:   # For the next layers we calculate (Output of previous layer)*W + b
                net = tf.matmul(result, self.get_weights_without_biases(layer_number)) + self.get_biases(layer_number)
            # we apply the Activation function on the 'net'
            result = self.activations[layer_number](net)
        return result



    def train(self, X_train, y_train, batch_size, num_epochs, alpha=0.8, regularization_coeff=1e-6):
        """
         Given a batch of data, and the necessary hyperparameters,
         this function trains the neural network by adjusting the weights and biases of all the layers.
         :param X: Array of input [n_samples,input_dimensions]
         :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
         the desired (true) class.
         :param batch_size: number of samples in a batch
         :param num_epochs: Number of times training should be repeated over all input data
         :param alpha: Learning rate
         :param regularization_coeff: regularization coefficient
         :return: None
         """
        for _ in range(num_epochs): # Run till specified number of epochs
            with tf.GradientTape(persistent = True) as tape: # Create a GradientTape to calculate partial derivates of loss w.r.t weights, biases
                # Create batches of data
                dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
                dataset = dataset.batch(batch_size)

                for X,y in dataset: # Update weights, biases for each batch
                    y_hat = self.predict(X)     # Check how model is predicting at present.
                    loss = self.cross_entropy_loss(y, y_hat)    # Calculate its loss
                    for layer_number in range(len(self.weights)):   # For each layer update the weights , biases
                        dloss_dw, dloss_db = tape.gradient(loss,[self.get_weights_without_biases(layer_number), self.get_biases(layer_number)]) # calculate the partial derivate of loss with respect to that layer's weights and bias
                        self.get_weights_without_biases(layer_number).assign_sub(alpha*dloss_dw)    # Update the weights
                        self.get_biases(layer_number).assign_sub(alpha*dloss_db)    # Update the bias




    def calculate_percent_error(self, X, y):
        """
        Given input samples and corresponding desired (true) output as indexes,
        this method calculates the percent error.
        For each input sample, if the predicted class output is not the same as the desired class,
        then it is considered one error. Percent error is number_of_errors/ number_of_samples.
        :param X: Array of input [n_samples,input_dimensions]
        :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
        the desired (true) class.
        :return percent_error
        """
        # Since the number of rows in X is the number of samples
        number_of_samples = X.shape[0]

        # Max index in each column is the predicted label
        y_hat = tf.argmax(self.predict(X),axis=1)

        # If prediction is same as target (y - y_hat at that index will be 0). If not same, its not 0
        incorrect_predictions = y - y_hat

        # Total number of non zero elements is the number of errors
        number_of_errors = len(incorrect_predictions[incorrect_predictions != 0])
        return (number_of_errors/number_of_samples)


    def calculate_confusion_matrix(self, X, y):
        """
        Given input samples and corresponding desired (true) output as indexes,
        this method calculates the confusion matrix.
        :param X: Array of input [n_samples,input_dimensions]
        :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
        the desired (true) class.
        :return confusion_matrix[number_of_classes,number_of_classes].
        Confusion matrix should be shown as the number of times that
        an image of class n is classified as class m where 1<=n,m<=number_of_classes.
        """
        y_hat = tf.argmax(self.predict(X), axis=1)

        """Approach - 1"""
        # To try Approach-1: Un-comment the below 4 lines  and comment out Approach-2
        # confusion_matrix = tf.zeros(shape = (10,10)).numpy()
        # for index in range(len(y)):
        #     confusion_matrix[y_hat[index]][y[index]]+= 1
        # return confusion_matrix

        """Approach - 2"""
        return tf.math.confusion_matrix(y_hat, y)


if __name__ == "__main__":
    from tensorflow.keras.datasets import mnist

    np.random.seed(seed=1)
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    # Reshape and Normalize data
    X_train = X_train.reshape(-1, 784).astype(np.float64) / 255.0 - 0.5
    y_train = y_train.flatten().astype(np.int32)
    input_dimension = X_train.shape[1]
    indices = list(range(X_train.shape[0]))
    # np.random.shuffle(indices)
    number_of_samples_to_use = 500
    X_train = X_train[indices[:number_of_samples_to_use]]
    y_train = y_train[indices[:number_of_samples_to_use]]
    multi_nn = MultiNN(input_dimension)
    number_of_classes = 10
    activations_list = [multi_nn.sigmoid, multi_nn.sigmoid, multi_nn.linear]
    number_of_neurons_list = [50, 20, number_of_classes]
    for layer_number in range(len(activations_list)):
        multi_nn.add_layer(number_of_neurons_list[layer_number], activation_function=activations_list[layer_number])
    for layer_number in range(len(multi_nn.weights)):
        W = multi_nn.get_weights_without_biases(layer_number)
        W = tf.Variable((np.random.randn(*W.shape)) * 0.1, trainable=True)
        multi_nn.set_weights_without_biases(W, layer_number)
        b = multi_nn.get_biases(layer_number=layer_number)
        b = tf.Variable(np.zeros(b.shape) * 0, trainable=True)
        multi_nn.set_biases(b, layer_number)
    multi_nn.set_loss_function(multi_nn.cross_entropy_loss)
    percent_error = []
    for k in range(10):
        multi_nn.train(X_train, y_train, batch_size=100, num_epochs=20, alpha=0.8)
        percent_error.append(multi_nn.calculate_percent_error(X_train, y_train))
    confusion_matrix = multi_nn.calculate_confusion_matrix(X_train, y_train)
    print("Percent error: ", np.array2string(np.array(percent_error), separator=","))
    print("************* Confusion Matrix ***************\n", np.array2string(np.array(confusion_matrix), separator=","))  # I added np.array() function on my confusion matrix.
