DELTA is the difference that should be added to the weight so we can say that:
DELTA = learning_rate * error_term_for_neuron * output

why delta? because we want to calculate the gradient of our relations so we can minimize the error and reach a local zero

so
for the output layer
error_term = output * (1 - output) * (expected - output)

hidden layers
