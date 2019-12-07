import numpy as np

class Perceptron():
    ''' Perceptron

        Concepts:
            Single perceptron ~ linear regression
            Multiple layer  - multiple decision regions

            f(x) = 1, if wx+b >0
            f(x) = 0, if wx+b <=0

            wx is a vector product:

    '''
    def __init__(self):
        ## initialize synaptic weights randomly with mean 0 to create weight matrix
        # seed random numbers
        np.random.seed(42)
        self.weights = 2 * np.random.random((3,1)) - 1

        print(f"Initial weights: \n{self.weights}")

    def activation_function(self, x):
        # activation function: the activation
        return 1 / (1 + np.exp(-x))

    def adjustment_function(self, x):
        # adjust synaptic weights: sigmoid derivative
        return x * (1 - x)

    def calc_outputs(self, input_layer):
        # Calculate weigth sum of inputs
        sum_weighted_inputs = np.dot(self.inputs, self.weights)

        # Calculate outputs:
        outputs = self.activation_function(sum_weighted_inputs)

        return outputs

    def train(self,X_train, y_train, epochs):
        # make input vector:
        self.inputs = X_train.astype(float)

        # Train the Perceptron:
        for iteration in range(epochs):
            self.outputs = self.calc_outputs(self.inputs)

            # Calculate error (simple diff):
            self.error = y_train - self.outputs

            # Calculate weights adjusment:
            adjustments = self.error * self.adjustment_function(self.outputs)

            # Update weights
            self.weights += np.dot(self.inputs.T, adjustments)

        return {
            'Weights': self.weights,
            'Outputs': self.outputs,
            'Error'  : self.error
        }

if __name__ == "__main__":
    ### Prepare the data:
    X_train = np.array([[0,0,1],
                        [1,1,1], # 1
                        [1,0,1], # 1
                        [0,1,1],
    ])

    y_train = np.array([[0,1,1,0]]).T


    ### Train the Perceptron
    perceptron = Perceptron()

    epochs = 100000
    params = perceptron.train(X_train, y_train, epochs)

    ### Display results:
    print('\nValues after training:')
    for name, value in params.items():
        print(f'{name}: \n{value}\n')

    ### predict:
    X_pred = []

