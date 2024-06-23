from line_profiler import LineProfiler
import numpy as np
import linear_regression 

def main():
    inputs = np.random.rand(500, 10)  # dati di esempio
    targets = np.random.rand(500, 1)  # target di esempio

    modello = linear_regression.m2()
    modello.run(inputs, targets)
    modello.test(inputs, targets)

if __name__ == "__main__":
    profiler = LineProfiler()
    profiler.add_function(linear_regression.m2.run)
    profiler.add_function(linear_regression.m2.test)
    profiler.add_function(linear_regression.A3.compute_x)
    profiler.add_function(linear_regression.A3.qr_decomposition_tall)

    profiler_wrapper = profiler(main)
    profiler_wrapper()

    profiler.print_stats()
