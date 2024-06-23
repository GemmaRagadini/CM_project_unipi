import numpy as np

class gradient: 
    def __init__(self, topology):
        self.hiddens = []
        self.hid_bias = []
        for i in range(len(topology)-2):
            self.hiddens.append(np.zeros((topology[i],topology[i+1]))) 
            self.hid_bias.append(np.full(topology[i+1], 0))
        # per l'output layer 
        self.out = np.zeros((topology[-2], topology[-1])) 
        self.out_bias = np.full(topology[-1], 0)    
