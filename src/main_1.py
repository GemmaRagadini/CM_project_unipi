import numpy as np
import sys
from m1 import neural_network
import activation_functions
import loss
import utilities
import regularizers

# Verifica che ci siano abbastanza argomenti
if len(sys.argv) < 5:
    print("Error: python main.py <algorithm_type> <alg_file> <output_file> <config_nn_file>")
    sys.exit(1)

# Tipo di algoritmo (1 o 2)
algorithm_type = sys.argv[1]

# File di configurazione dell'algoritmo
alg_file = sys.argv[2]

# File di output per salvare i risultati
output_file = sys.argv[3]

# file config rete neurale
config_nn_file = sys.argv[4]

# Lettura del file di configurazione della rete neurale
(training_set, topology, epochs, minibatch_size, reg_coef) = utilities.read_config_nn(config_nn_file)

# Lettura del dataset di addestramento
(inputs, targets) = utilities.read_csv(training_set)

# Impostazione della topologia della rete neurale
topology = np.insert(topology, 0, len(inputs[0])) 
if isinstance(targets[0], np.ndarray):
    topology = np.append(topology, len(targets[0]))
else:
    topology = np.append(topology, 1)

# Impostazione del regularizzatore L2
reg = regularizers.L2(reg_coef)

# Creazione della rete neurale
act_hidden = activation_functions.relu
act_output = activation_functions.linear
loss_function = loss.least_squared
net = neural_network(topology, act_hidden, act_output, loss_function, reg)

# Impostazione dell'algoritmo di addestramento usando il tipo e il file passati come argomenti
net.set_algorithm(algorithm_type, alg_file)

# Addestramento della rete neurale
loss_tot = net.run_training(inputs, targets, epochs, minibatch_size)

# Salvataggio dei pesi finali in un file .npy
np.save(output_file, loss_tot)