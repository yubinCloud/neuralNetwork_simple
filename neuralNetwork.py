import numpy
import scipy.special
import imageio
import matplotlib.pyplot


# neural network class definition
class NeuralNetwork:
    # initialise the neural network
    def __init__(self, inputNodes=784, hiddenNodes=100, outputNodes=10, learningRate=0.3):
        # set number of nodes in each input, hidden, output layer
        self.iNodes = inputNodes
        self.hNodes = hiddenNodes
        self.oNodes = outputNodes
        # learning rate
        self.lr = learningRate
        # link weight matrices, wih and who
        # weights inside the arrays are w_i_j, where link is from node i to node j in the next layer
        # w11 w21
        # w12 w22 etc
        self.wih = numpy.random.normal(0.0, pow(self.hNodes, -0.5), (self.hNodes, self.iNodes))
        self.who = numpy.random.normal(0.0, pow(self.oNodes, -0.5), (self.oNodes, self.hNodes))
        # activation function is the sigmoid function
        self.activationFunction = lambda x: scipy.special.expit(x)

    # train the neural network
    def train(self, inputs_list, targets_list):
        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activationFunction(hidden_inputs)
        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activationFunction(final_inputs)
        # error is the (target - actual)
        output_errors = targets - final_outputs
        # hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden_errors = numpy.dot(self.who.T, output_errors)
        # update the weights for the links between the hidden and output layers
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                        numpy.transpose(hidden_outputs))
        # update the weights for the links between the input and hidden layers
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                                        numpy.transpose(inputs))

    # query the neural network
    def query(self, inputs_list):
        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activationFunction(hidden_inputs)
        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activationFunction(final_inputs)

        return final_outputs

    # get the parameter of this object
    def getParameter(self):
        return {'iNodes': self.iNodes,
                'hNodes': self.hNodes,
                'oNodes': self.oNodes,
                'learning_rate': self.lr,
                'wih': self.wih,
                'who': self.who,
                'activationFunction': self.activationFunction}


if __name__ == '__main__':
    # number of input, hidden and output nodes
    input_nodes = 784
    hidden_nodes = 100
    output_nodes = 10
    # learing rate is 0.3
    learning_rate = 0.3

    # create instance of NeuralNetwork
    n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

    # load the minist training data CSV file into a list
    training_data_file = open("mnist_train_2000.csv", 'r', encoding='utf-8')
    training_data_list = training_data_file.readlines()
    training_data_file.close()

    # train the neural network
    print('Training...')

    # go through all records in the training data set
    for record in training_data_list:
        # split the record by the ',' commas
        all_values = record.split(',')
        # scale and shift the inputs
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        # create the target output values (all 0.01, except the desired label which is 0.99)
        targets = numpy.zeros(output_nodes) + 0.01
        # all_values[0] is the target label for this record
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)

    # load the mnist test data CSV file into a list
    test_data_file = open("mnist_test_1000.csv", 'r', encoding='utf-8')
    test_data_list = test_data_file.readlines()
    test_data_file.close()

    # test the neural network
    print('Testing...')
    # scorecard for how well the network performs, initially empty
    scorecard = list()

    # go through all the records in the test data set
    for record in test_data_list:
        # split the record by the ',' commas
        all_values = record.split(',')
        # correct answer is the first value
        correct_label = int(all_values[0])
        print("correct label: ", correct_label)
        # scale and shift the inputs
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        # query the network
        outputs = n.query(inputs)
        # the index of the highest value corresponds to the label
        label = numpy.argmax(outputs)
        print("network's answer: ", label)
        # append correct or incorrect to list
        if label == correct_label:
            scorecard.append(1)
        else:
            scorecard.append(0)

    # calculate the performance score, the fraction of correct ansers
    scorecard_array = numpy.asarray(scorecard)
    print("performance = ", scorecard_array.sum() / scorecard_array.size)
