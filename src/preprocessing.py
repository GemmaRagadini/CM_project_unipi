def oneHotEncoding(input):
    if len(input) != 6:
        print("preprocessing:oneHotEncoding:ERROR")
        exit()
    encoded = []
    for i in range(len(input)):
        if i == 0 or i == 1 or i == 3:
            if input[i] == 1: 
                encoded+=[0,0,1]
            if input[i] == 2:
                encoded+=[0,1,0]
            if input[i] == 3:
                encoded+=[1,0,0]
        if i == 2 or i == 5:
            if input[i] == 1:
                encoded+=[0,1]
            if input[i] == 2:
                encoded+=[1,0]
        if i == 4:
            if input[i] == 1:
                encoded+=[0,0,0,1]
            if input[i] == 2:
                encoded+=[0,0,1,0]
            if input[i] == 3:
                encoded+=[0,1,0,0]
            if input[i] == 4:
                encoded+=[1,0,0,0]
    return encoded   


def normalize(inputs):
    num_features = len(inputs[0])
    num_righe = len(inputs)
    
    inputs_normalizzati = []
    for i in range(num_features):
        colonna = [riga[i] for riga in inputs]
        valore_min = min(colonna)
        valore_max = max(colonna)
        inputs_normalizzati.append([(valore - valore_min) / (valore_max - valore_min) for valore in colonna])
    
    inputs_normalizzati_trasposti = list(map(list, zip(*inputs_normalizzati))) 
    return inputs_normalizzati_trasposti
