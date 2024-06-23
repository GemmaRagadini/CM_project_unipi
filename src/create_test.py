import pandas as pd

# Carica il dataset
file_path = "../data/Diabetes.csv"
dataset = pd.read_csv(file_path)

# Salva la prima riga per includerla nel nuovo dataset
prima_riga = dataset.iloc[0]

# Escludi la prima riga dall'estrazione casuale per il test
dataset = dataset.iloc[1:]

# Percentuale di dati da utilizzare per il test
percentuale_test = 0.2  # Puoi regolare questa percentuale secondo le tue esigenze

# Calcola il numero di righe da estrarre per il test
num_righe_test = int(len(dataset) * percentuale_test)

# Estrai casualmente gli indici delle righe per il test
indici_test = dataset.sample(n=num_righe_test, random_state=42).index

# Estrai il test set dal dataset originale utilizzando gli indici
test_set = dataset.loc[indici_test]

# Rimuovi le righe del test set dal dataset originale
dataset = dataset.drop(indici_test)

# Concatena la prima riga con il test set
test_set = pd.concat([prima_riga.to_frame().transpose(), test_set])

# # Crea il dataset di addestramento escludendo semplicemente il primo elemento
# training_set = dataset.iloc[1:]

dataset.to_csv("../data/new_diabetes.csv", index=False)

# Scrivi il test set in un nuovo file CSV
test_file_path = "../data/Diabetes_test.csv"
test_set.to_csv(test_file_path, index=False)




# Ora puoi utilizzare training_set per addestrare la tua rete neurale
