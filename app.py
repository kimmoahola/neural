from numpy import random
from numpy.ma import exp, array, dot


class NeuralNetwork:
    def __init__(self):
        # Seed the random number generator, so it generates the same numbers
        # every time the program runs.
        random.seed(1)

        # We model a single neuron, with 3 input connections and 1 output connection.
        # We assign random weights to a 3 x 1 matrix, with values in the range -1 to 1
        # and mean 0.
        self.synaptic_weights = 2 * random.random((3, 5)) - 1

    # The Sigmoid function, which describes an S shaped curve.
    # We pass the weighted sum of the inputs through this function to
    # normalise them between 0 and 1.
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    # The derivative of the Sigmoid function.
    # This is the gradient of the Sigmoid curve.
    # It indicates how confident we are about the existing weight.
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    # We train the neural network through a process of trial and error.
    # Adjusting the synaptic weights each time.
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in range(number_of_training_iterations):
            # Pass the training set through our neural network (a single neuron).
            output = self.think(training_set_inputs)

            # Calculate the error (The difference between the desired output
            # and the predicted output).
            # print(training_set_outputs)
            # print(output)
            error = training_set_outputs - output
            # print(error)
            # print('----')

            # Multiply the error by the input and again by the gradient of the Sigmoid curve.
            # This means less confident weights are adjusted more.
            # This means inputs, which are zero, do not cause changes to the weights.
            adjustment = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))

            # Adjust the weights.
            self.synaptic_weights += adjustment

    # The neural network thinks.
    def think(self, inputs):
        # Pass inputs through our neural network (our single neuron).
        return self.__sigmoid(dot(inputs, self.synaptic_weights))


mode_off = [1, 0, 0, 0, 0]
mode_heat8 = [0, 1, 0, 0, 0]
mode_heat10 = [0, 0, 1, 0, 0]
mode_heat16 = [0, 0, 0, 1, 0]
mode_heat20 = [0, 0, 0, 0, 1]

output_groups = ['off', 'heat8', 'heat10', 'heat16', 'heat20']

# tavoite
# virhe
# virhe muutosnopeus
# sisä miinus ulko
# sisä miinus 12h ulkolämpötilaennuste
# PID-säätimen i-termi
# oliko edellinen komento lämmitys

training_set_inputs = array([
    [5, 1, 0],
    [5, 1, 1],
    [8, 0, 0],
    [5, -1.5, 0],
    # [5, 2, 0, 10, 10, 0, 1],
    # [5, 1.2, 0, 10, 10, 0, 1],
    # [5, 0.5, 0, 10, 10, 0, 1],
    # [5, 0.5, -0.1, 10, 10, 0, 1],
    # [5, -1, 0, 10, 10, 0, 1],
    # [7, 0, 0, 20, 18, 0, 1],
    # [7, 1, 0, 20, 18, 1, 1],
    # [5, -0.9, 0, 11, 7, -1, 1],
    # [5, -1.2, 0, 11, 7, -1.5, 1],
    # [5, -0.1, 0, 11, 7, -0.5, 0],
])

# training_set_outputs = array([[1, 1, 0]]).T
training_set_outputs = array([
    mode_heat8,
    mode_heat16,
    mode_heat10,
    mode_off,
    # mode_heat16,
    # mode_heat10,
    # mode_heat8,
    # mode_off,
    # mode_heat16,
    # mode_heat20,
    # mode_heat8,
    # mode_off,
    # mode_off,
])


def test(neural_network):
    test_sets = [
        ([4, 0.5, 0.1], mode_heat8),
        # ([6,  0,    0.5,   15, 10, 1, 1], mode_heat16),
        # ([5,  2,    0,   10, 10,  0, 1], mode_heat16),
        # ([5, -1.2,  0,   11,  7, -1.5, 1], mode_off),
        # ([5, -0.1,  0,   11,  7, -0.5, 0], mode_off),
        # ([5, -0.9,  0,   11,  7, -1, 1], mode_heat8),
    ]

    failed_count = 0

    def run_test(i, c):
        result = neural_network.think(array(i))
        max_result = max(result)
        output_group_index = result.tolist().index(max_result)
        test_output = output_groups[output_group_index]
        correct_output_text = output_groups[c.index(max(c))]

        if test_output != correct_output_text:
            print('\ntest did not pass')
            print('test_input', test_input)
            print('test output', test_output)
            print('correct', correct_output_text)
            print('result', result)
            return 1
        return 0

    for test_input, correct_output in zip(training_set_inputs.tolist(), training_set_outputs.tolist()):
        failed_count += run_test(test_input, correct_output)

    for test_input, correct_output in test_sets:
        failed_count += run_test(test_input, correct_output)

    total_count = len(test_sets) + len(training_set_inputs)
    success_count = total_count - failed_count
    print(f'Accuracy {success_count}/{total_count} {success_count / total_count * 100} %')


def main():

    # Intialise a single neuron neural network.
    neural_network = NeuralNetwork()

    print("Random starting synaptic weights: ")
    print(neural_network.synaptic_weights)

    # Train the neural network using a training set.
    # Do it 10,000 times and make small adjustments each time.
    neural_network.train(training_set_inputs, training_set_outputs, 1000)

    print("New synaptic weights after training: ")
    print(neural_network.synaptic_weights)

    test(neural_network)


if __name__ == "__main__":
    main()
    #
    # # Test the neural network with a new situation.
    # print("Considering new situation -> ?: ")
    #
    # result = neural_network.think(array([6,  0,    0.5,   15, 10, 1, 1]))
    #
    # print(result)
    # max_result = max(result)
    # output_group_index = result.tolist().index(max_result)
    # print(output_groups[output_group_index])
    #
    # # if result[0] < 0.5:
    # #     print('no heat')
    # # else:
    # #     print('heat')
