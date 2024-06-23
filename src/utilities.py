import numpy as np 
import preprocessing 
import matplotlib.pyplot as plt
import re
import csv
import sys
from scipy.optimize import curve_fit
from skimage.transform import radon, iradon
import os

def read_config_nn(filename):
    
    tr = None
    topology = None
    epochs = 0
    minibatch_size = 0
    reg_coefficient = 0

    with open(filename, "r") as file:
        for line in file:
            match = re.match(r'^(\w+)\s*=\s*(.*)$', line)
            if match:
                key = match.group(1)
                value = match.group(2).strip()
                # Assegna i valori alle variabili in base alla chiave
                if key == "TR":
                    tr = value
                elif key == "topology":
                    topology = [int(x) for x in value.split(",")]
                elif key == "epochs":
                    epochs = int(value)
                elif key == "minibatch_size":
                    minibatch_size = int(value)
                elif key == "reg_coefficient":
                    reg_coefficient = float(value)
                

    return tr, topology, epochs, minibatch_size, reg_coefficient


# def read_config_a1(filename):
#     stepsize = 0
#     momentum = 0
#     with open(filename, "r") as file:
#         for line in file:
#             match = re.match(r'^(\w+)\s*=\s*(.*)$', line)
#             if match:
#                 key = match.group(1)
#                 value = match.group(2).strip()
#                 if key == "stepsize":
#                     stepsize = float(value)
#                 elif key == "momentum":
#                     momentum = float(value)
#     return stepsize, momentum


def read_config_alg(filename):      
    stepsize = 0
    momentum = 0
    k = 0
    beta = 0
    with open(filename, "r") as file:
        for line in file:
            match = re.match(r'^(\w+)\s*=\s*(.*)$', line)
            if match:
                key = match.group(1)
                value = match.group(2).strip()
                if key == "k":
                    k = float(value)
                elif key == "beta":
                    beta = float(value)
                elif key == "stepsize":
                    stepsize = float(value)
                elif key == "momentum":
                    momentum = float(value)
    return k, beta, stepsize, momentum


# legge da file per ML-CUP
def read_csv(filename):
    inputs = []
    targets = []
    numero_di_riga_iniziale = 7 
    with open(filename, 'r') as file:
        lettore_csv = csv.reader(file)
        tutte_le_righe = list(lettore_csv)
        # Leggi solo le righe dalla posizione desiderata in avanti
        righe_selezionate = tutte_le_righe[numero_di_riga_iniziale:]
        righe_selezionate = np.array(righe_selezionate)
        righe_selezionate = righe_selezionate.astype(float)
        # divisione dei dati in input , output e il primo valore ignorato
        for riga in righe_selezionate:
            inputs.append(riga[1:11])
            targets.append(riga[-3:])
    return (inputs,targets)



def read_monk(filename):
    targets = []
    inputs = []
    with open(filename, 'r') as file:
        for line in file:
            # Dividi la riga in una lista di stringhe
            values = line.split()
            # Aggiungi il primo numero a "target"
            targets.append(int(values[0]))
            # Aggiungi i successivi valori a "inputs" come lista di interi
            input_features = list(map(int, values[1:-1]))
            inputs.append(input_features)
    encoded_input = []
    for i in range(len(inputs)):
        encoded_input.append(preprocessing.oneHotEncoding(inputs[i]))
    return(np.array(encoded_input), np.array(targets))



def avg_for_layer(minibatch_outputs):  
    avg = []
    for i in range(len(minibatch_outputs[0])):
        minibatch_outputs[0][i] = np.ravel(minibatch_outputs[0][i])
        avg.append(np.zeros_like(minibatch_outputs[0][i]))
        for j in range(len(minibatch_outputs)):
            avg[-1] = np.add(avg[-1],minibatch_outputs[j][i])
        avg[-1]= np.divide(avg[-1],len(minibatch_outputs))

    return avg

def compute_accuracy(outputs, targets):
    outputs = [1 if valore >= 0.5 else 0 for valore in np.ravel(outputs)]
    accuracy = 0
    for i in range(len(targets)):
        if outputs[i] == targets[i]:
            accuracy+=1
    accuracy/=len(targets)
    return accuracy*100
    

# trasforma un array di valori in un array di 0/1
def bin_transform(x):
    return np.where(x > 0.5, 1, 0)


def plot_error(x, lambda_value=None, filename='plot.png'):

    plt.figure(figsize=(10, 6))
    plt.plot(x, linewidth=3, color='b')
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel("log(Er)", fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.grid(True)
    plt.show()


def f(x,a,o,n,x0,b):
    return a/((x-o)**n) + x0 + b*np.exp(-x) 

def fit(loss_values):
    x = np.arange(1,len(loss_values)+1)
    popt,pcov = curve_fit(f,x,loss_values, p0 = [1,-5,2,0, 1])
    print(popt)
    print(np.sqrt(pcov[3][3]))
    return f(x,popt[0], popt[1], popt[2], popt[3], popt[4])

def is_upper_triangular(matrix):
    return np.allclose(matrix, np.triu(matrix))

def is_orthogonal(matrix):
    return np.allclose(np.dot(matrix, np.transpose(matrix)), np.eye(len(matrix)))

def is_submatrix_of_orthogonal_matrix(A):
    # Calcola la matrice di Gram
    G = np.dot(np.transpose(A), A)
    
    # Crea una matrice identità della dimensione appropriata
    I = np.eye(A.shape[1])
    
    # Verifica se G è uguale alla matrice identità
    return np.allclose(G, I)

def plot_conv_rate(loss, opt_loss, p_values=[1,1.2,1.5,1.7,2]):
    plt.figure(figsize=(10, 6))
    for p in p_values:
        rate = []
        for i in range(len(loss)-1):
            numerator = loss[i+1] - opt_loss
            denominator = (loss[i] - opt_loss)**p
            rate.append(numerator / denominator)

        plt.plot(rate, linewidth=2, label=f'p = {p}')
    
    plt.title('Convergence rate', fontsize=14)
    plt.xticks(fontsize=13)  # Imposta la dimensione del testo sull'asse x
    plt.yticks(fontsize=13)  # Imposta la dimensione del testo sull'asse y
    plt.xlabel('Epochs', fontsize=13)
    plt.ylabel('Log(Rate)', fontsize=13)
    plt.legend(fontsize=13)
    plt.grid(True)
    plt.show()


# calcola il condition number di una matrice A attraverso i valori singolari 
def cond_number(A):
    U, S, Vt = np.linalg.svd(A)
    sigma_max = np.max(S)
    sigma_min = np.min(S[S > 0])  # escludo quelli nulli 
    kappa = sigma_max / sigma_min
    return kappa