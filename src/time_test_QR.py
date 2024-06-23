import linear_regression
import utilities
import matplotlib.pyplot as plt 
import numpy as np 

solver = linear_regression.m2()

# TEST TEMPO AL VARIARE DI m
# Lista dei numeri di righe da testare
m_values = range(10, 800, 10)
times = []

# Misura i tempi di esecuzione della decomposizione per diverse dimensioni di m
for m in m_values:
    elapsed_time = solver.measure_time(m)
    times.append(elapsed_time)
    print(f"m={m}, Time={elapsed_time:.4f} seconds")

# plot dei tempi di esecuzione
# Grafico del tempo di esecuzione in funzione di m
plt.plot(m_values, times, marker='o')
plt.xlabel('Number of rows', fontsize=14)
plt.ylabel('Execution time', fontsize=14)
plt.grid(True)
plt.show()

