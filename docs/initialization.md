# Xavier Weight Initialization
The xavier initialization method is calculated as a random number with a uniform probability distribution (U) between the range -(1/sqrt(n)) and 1/sqrt(n), where n is the number of inputs to the node.

weight = U [-(1/sqrt(n)), 1/sqrt(n)]

# He Weight Initialization
The he initialization method is calculated as a random number with a Gaussian probability distribution (G) with a mean of 0.0 and a standard deviation of sqrt(2/n), where n is the number of inputs to the node.

weight = G (0.0, sqrt(2/n))

thx2: https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks 