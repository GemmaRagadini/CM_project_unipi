import loss
import numpy as np
import utilities
from weights_matrices import weigths_matrices
from gradient import gradient
import sys

opt_loss = 0.11163446915853614

class neural_network:

    # layer sizes passato dall'esterno
    def __init__(self, topology, activationFunctionForHidden, activationFunctionForOutput, lossFunction, regularizer):
        
        self.act_hidden = activationFunctionForHidden
        self.act_output = activationFunctionForOutput
        self.lossFunction = lossFunction
        # lista di neuroni per ogni layer
        self.topology = topology 
        # tipo di regolarizzazione
        self.regularizer = regularizer
        # matrici dei pesi
        self.weigths = weigths_matrices(self.topology) 
        self.min_loss = sys.maxsize      

    def set_algorithm(self, algorithm_type, file_alg):

        if int(algorithm_type) == 1:
            from a1 import A1
            self.algorithm = A1(self,file_alg)
        if int(algorithm_type) == 2:
            from a2 import A2  
            self.algorithm = A2(self,file_alg)  

    # propagazione in avanti di un singolo esempio
    def forwardpropagation(self, data):
        hidden_outputs = []
        # hidden layers
        for i in range(len(self.weigths.hiddens)):
            if i == 0:
                hidden_outputs.append(np.ravel(self.act_hidden(np.dot(data, self.weigths.hiddens[i]) + self.weigths.hid_bias[i])))
            else: 
                hidden_outputs.append(np.ravel(self.act_hidden(np.dot(hidden_outputs[i-1], self.weigths.hiddens[i]) + self.weigths.hid_bias[i])))
        final_output = np.ravel(self.act_output(np.dot(hidden_outputs[-1], self.weigths.out) + self.weigths.out_bias))
        # final_output è l'output dell'output layer , hidden_outputs è l'array degli output di ogni hidden layer
        return (hidden_outputs,final_output) 
    

    # calcola la forward propagation per un minibatch e la media
    def compute_minibatch(self, minibatch_data):
        hidden_outputs = []
        final_outputs = []
        for input_data in minibatch_data:
            outputs = self.forwardpropagation(input_data)
            hidden_outputs.append(outputs[0])
            final_outputs.append(outputs[1])
        return (utilities.avg_for_layer(hidden_outputs), np.divide(final_outputs,len(minibatch_data)) )



    def run_training(self, tr_data, tr_targets, numberEpochs, minibatch_size):
        loss_tot = []
        # creo gradiente vuoto
        grad = gradient(self.topology)
        # Addestramento del modello
        for epoch in range(numberEpochs):
            print("Epoca: ", epoch)
            epoch_loss_sum = 0
            for i in range(int(len(tr_data)/minibatch_size)):

                minibatch_data  = tr_data[i*minibatch_size:i*minibatch_size + minibatch_size]
                minibatch_target = tr_targets[i*minibatch_size:i*minibatch_size + minibatch_size]
                (avg_hidden_outputs, avg_final_output) = self.compute_minibatch(minibatch_data)
                avg_loss = self.lossFunction(np.divide(minibatch_target,len(minibatch_target)),avg_final_output)
                if self.regularizer:
                    avg_loss += self.regularizer(self.weigths.out) 
                    for j in range(len(self.weigths.hiddens)):
                        avg_loss += self.regularizer(self.weigths.hiddens[j])
                    

                d_loss = loss.derivative(self.lossFunction)(np.divide(minibatch_target, len(minibatch_target)),avg_final_output)
                if len(loss_tot) == 0:
                    epoch_loss = sys.maxsize
                else:
                    epoch_loss = loss_tot[-1]
              
                grad = self.algorithm.learning(d_loss, avg_hidden_outputs,  np.average(minibatch_data,axis=0) , grad , epoch_loss)
                # somma della loss per ogni esempio per poi farne la media dell'epoca
                epoch_loss_sum += avg_loss 

            for i in range(len(tr_data)%minibatch_size):
                if i == 0:
                    minibatch_size+=1 # per calcolare la media
                minibatch_data  = tr_data[-i:]
                minibatch_target = tr_targets[-i:]
                (avg_hidden_outputs, avg_final_output) = self.compute_minibatch(minibatch_data)
                avg_loss = self.lossFunction(np.divide(minibatch_target,len(minibatch_target)),avg_final_output)            
                if self.regularizer:
                    avg_loss += self.regularizer(self.weigths.out) 
                    for j in range(len(self.weigths.hiddens)):
                        avg_loss += self.regularizer(self.weigths.hiddens[j])

               
                d_loss = loss.derivative(self.lossFunction)(np.divide(minibatch_target, len(minibatch_target)),avg_final_output)
                print(d_loss)
                epoch_loss = loss_tot[-1]
                grad = self.algorithm.learning(d_loss, avg_hidden_outputs,np.average(minibatch_data,axis=0),grad , epoch_loss)
                epoch_loss_sum += avg_loss 

            # loss media per ogni epoca
            loss_tot.append(epoch_loss_sum/len(tr_data))
                
            print(loss_tot[-1])
            if loss_tot[-1] < self.min_loss:
                self.min_loss = loss_tot[-1]

        # salva su file txt 
        np.savetxt("loss_per_epochs.txt", np.reshape(loss_tot,(-1, 1)), fmt='%.14f')
        print(self.min_loss)
        utilities.plot_error(np.log(np.divide(np.subtract(loss_tot,opt_loss), opt_loss)))
        # utilities.plot_conv_rate(loss_tot,opt_loss)
    
        # return self.weigths.get_weights_as_vector()
        return loss_tot

    # per la tomografia 
    def compute_loss(self, x,y):
        loss_values = []
        for i in range(len(x)):
            out = self.forwardpropagation(x[i])[1]    
            loss_values.append(self.lossFunction(out,y[i]))
 
        return np.average(loss_values)
  