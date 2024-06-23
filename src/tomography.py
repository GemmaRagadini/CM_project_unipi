import sys
import matplotlib.pyplot as plt
from skimage.transform import radon, iradon
import regularizers
import activation_functions
import loss
from m1 import neural_network
import utilities
import numpy as np


def plot_loss_tomography(net, x, y, t_min, t_max):

    t = np.ravel(np.linspace(t_min, t_max, 100))

    original_weights = net.weigths.get_weights_as_vector()
    direction = np.random.uniform(-1,1,len(original_weights))
    direction = direction / np.linalg.norm(direction)
    loss_values = []
    for t_val in t:
        new_weights = original_weights + t_val * direction
        net.weigths.set_weigths_from_vector(new_weights)
        loss = net.compute_loss(x, y)/len(x)
        loss_values.append(loss)

    plt.figure(figsize=(10, 6))
    plt.plot(t, loss_values, linewidth=3, color ='r', label='Loss Tomography')
    plt.xlabel('t', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.grid(True)
    plt.show()



(training_set,topology,epochs,minibatch_size,reg_coef) = utilities.read_config_nn("../config/config_nn.txt")
(inputs,targets) = utilities.read_csv(training_set)

# (training_set,test_set,topology,stepsize,epochs,minibatch_size,reg_coef,momentum,test) = utilities.read_config(config_file)
# (inputs,targets) = utilities.read_csv(training_set)
# l'input layer è della dimensione delle features dell'input
topology = np.insert(topology, 0, len(inputs[0])) 
# l'output layer è della dimensione delle features del target
if isinstance(targets[0], np.ndarray):
    topology = np.append(topology, len(targets[0]))
else:
    topology = np.append(topology,1)
reg = regularizers.L2(reg_coef)

act_hidden = activation_functions.relu
act_output = activation_functions.linear
loss_function = loss.least_squared
net = neural_network(topology, act_hidden, act_output, loss_function, reg)
# net = neural_network(topology, stepsize, act_hidden, act_output, loss_function, reg, momentum,1)
algorithm_type = sys.argv[1]
# A1 o A2
net.set_algorithm(algorithm_type)
net.run_training(inputs,targets,epochs,minibatch_size)
plot_loss_tomography(net, inputs, targets,-30, 30)


