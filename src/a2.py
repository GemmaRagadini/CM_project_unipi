import numpy as np
import activation_functions
from m1 import neural_network
from gradient import gradient
import sys
import utilities

class A2(neural_network):

    def __init__(self, nn : neural_network, filename):
        self.nn = nn    
        self.topology = nn.topology
        (self.k,self.beta,stepsize,momentum) = utilities.read_config_alg(filename)
        self.iteration = 0
        self.alpha = 0.5
        self.i = 1
        self.j = 1
        self.min_loss = sys.maxsize
      
        
    # subgradient method
    def learning(self, loss_deriv,  hidden_outputs, minibatch_data, deflection, epoch_loss):
        
        self.iteration += 1

        
        # update min loss
        if epoch_loss < self.min_loss:
            self.min_loss = epoch_loss
            # reset j
            self.j = 1
        else:
            self.j += 1

        if self.j >= self.k:
            self.j = 1
            self.i += 1

        subgrad_curr  = self.backpropagation(loss_deriv, hidden_outputs, minibatch_data)
        # output layer    
 
        self.alpha = self.compute_alpha(subgrad_curr.out, deflection.out)
        deflection.out = self.compute_deflection(subgrad_curr.out, deflection.out)

        stepsize = self.compute_stepsize_DSS()
                                            
        self.nn.weigths.out = self.update_weights(self.nn.weigths.out, stepsize, deflection.out)


    
        self.alpha = self.compute_alpha(subgrad_curr.out_bias, deflection.out_bias)
        deflection.out_bias = self.compute_deflection(subgrad_curr.out_bias, deflection.out_bias)
        stepsize = self.compute_stepsize_DSS()
        self.nn.weigths.out_bias = self.update_weights(self.nn.weigths.out_bias, stepsize, deflection.out_bias)

        #hidden layers
        for i in range(len(self.nn.weigths.hiddens)):  
        
            self.alpha = self.compute_alpha(subgrad_curr.hiddens[i], deflection.hiddens[i])
            deflection.hiddens[i] = self.compute_deflection(subgrad_curr.hiddens[i], deflection.hiddens[i])
            stepsize = self.compute_stepsize_DSS()
            self.nn.weigths.hiddens[i] = self.update_weights(self.nn.weigths.hiddens[i], stepsize, deflection.hiddens[i])

            self.alpha = self.compute_alpha(subgrad_curr.hid_bias[i], deflection.hid_bias[i])
            deflection.hid_bias[i] = self.compute_deflection(subgrad_curr.hid_bias[i], deflection.hid_bias[i])
            stepsize = self.compute_stepsize_DSS()
            self.nn.weigths.hid_bias[i] = self.update_weights(self.nn.weigths.hid_bias[i], stepsize, deflection.hid_bias[i])

        return deflection

    
    # nell'intervallo [0.1]
    def compute_alpha(self, g, d_prev):

        v = g-d_prev
      
        if v.all() == 0:
            alpha = 0.5
        else: 
            # prodotto elemento per elemento
            alpha = -(np.sum(v*d_prev))/np.sum(v**2)
            # valuto il valore agli estremi 
            if alpha < 0 or alpha > 1: 
                # se in 0 la funzione è minore che in 1
                if np.sum(d_prev**2) <= np.sum(v**2) + 2*np.sum(v*d_prev) + np.sum(d_prev**2):
                    alpha = 0.0001
                else:
                    alpha = 0.9999
        
        return alpha



    def compute_deflection(self , g, d):
        d = self.alpha*g + (1-self.alpha)*d
        return d


    def compute_stepsize_DSS(self): 
        return self.beta/self.i
 
    def update_weights(self, weigths, stepsize, deflection):
        weigths = weigths - stepsize*deflection
        return weigths

    def backpropagation(self, d_loss, hidden_outputs , data):

        # creo istanza classe gradiente 
        grad = gradient(self.nn.topology)
        # calcolo gradiente per output layer
        input = hidden_outputs[-1]
        next_weigths = self.nn.weigths.out
        delta_out = activation_functions.derivative(self.nn.act_output)(np.dot(input, next_weigths)) * d_loss
        grad.out = np.outer(input, delta_out) 
        grad.out_bias = delta_out
        
        delta_hiddens = []
        d_f = activation_functions.derivative(self.nn.act_hidden) 

        # gradiente per gli hidden layers
        if len(self.nn.weigths.hiddens) == 1: # se c'è un solo hidden layer 
            next_weigths = np.transpose(self.nn.weigths.out)
            delta_hiddens.append(np.dot(delta_out, next_weigths)*d_f(np.dot(data,self.nn.weigths.hiddens[0])))
            grad.hiddens.append(np.outer(data, delta_hiddens[-1]))
            grad.hid_bias.append(delta_hiddens[-1])

        else:
            for i in range(len(self.nn.weigths.hiddens)-1, -1, -1):
                
                delta = delta_out if i == len(self.nn.weigths.hiddens)-1 else delta_hiddens[-1]
                input = data if i == 0 else hidden_outputs[i-1]
                next_weigths = np.transpose(self.nn.weigths.out) if i == len(self.nn.weigths.hiddens)-1 else np.transpose(self.nn.weigths.hiddens[i+1])
                
                delta_hiddens.append(np.dot(delta, next_weigths)*d_f(np.dot(input,self.nn.weigths.hiddens[i])))
                grad.hiddens.append(np.outer(input, delta_hiddens[-1]))
                grad.hid_bias.append(delta_hiddens[-1])
        
        # inverto l'ordine delle matrici perché sono stati riempiti al contrario
        grad.hiddens = grad.hiddens[::-1]      
        grad.hid_bias = grad.hid_bias[::-1]

        return grad

    