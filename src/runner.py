# runner.py
import subprocess
import numpy as np
import os
import matplotlib.pyplot as plt

# Directory per i risultati
results_dir = 'results'
os.makedirs(results_dir, exist_ok=True)

# Lista dei file di configurazione degli algoritmi e dei tipi di algoritmi
alg_files = ['../config/runner/config1.txt', 
             '../config/runner/config2.txt', '../config/runner/config3.txt', '../config/runner/config4.txt', '../config/runner/config5.txt', '../config/runner/config6.txt', 
             '../config/runner/config7.txt', '../config/runner/config8.txt']
algorithm_types = ['1','1','1','1','2','2', '2', '2']


# esegue main.py più volte e salva i risultati 
for i in range(len(alg_files)):
    result_file = f"{results_dir}/result_{i+1}.npy"
    subprocess.run(['python3', 'main_1.py', algorithm_types[i] , alg_files[i], result_file, '../config/config_nn.txt'])

# Caricare e combinare i dati
all_data = []
for i in range(len(alg_files)):
    result_file = f"{results_dir}/result_{i+1}.npy"
    array_data = np.load(result_file)
    all_data.append(array_data)

# Convertire la lista in un array numpy per facilitare l'elaborazione
all_data = np.array(all_data)


# # Creazione della figura e degli assi
# plt.figure(figsize=(10, 6))

# # Plot di ogni serie di dati in all_data
# opt_loss = 0.11163446915853614
# for i, error_values in enumerate(all_data):
#     plt.plot(np.log(np.divide(np.subtract(error_values[30:],opt_loss), opt_loss)), linewidth=2, label=f'Config {i+1}')  # Etichettatura per ogni serie

# # Aggiunta delle etichette e della legenda
# plt.xlabel('Epochs', fontsize=14)
# plt.ylabel('Log(Er)', fontsize=14)
# # plt.title('Error Values Across Epochs for Different Models')
# plt.legend()
# plt.show()

# Calcolare la media degli array caricati
mean = np.mean(all_data, axis=0)

# Calcolare la norma di ogni array rispetto alla media e trovare la norma massima
norms = np.linalg.norm(all_data, axis=1)
max_norm = np.max(norms)

# Calcolare la distanza relativa al massimo di ogni array rispetto alla media
distances = np.linalg.norm(all_data - mean, axis=1) / max_norm
print("DISTANCES: ", distances)