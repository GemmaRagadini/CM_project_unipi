# main per linear regression 
import linear_regression
import utilities

(inputs, targets) = utilities.read_csv("../data/ML-CUP23-TR.csv")
solver = linear_regression.m2()
solver.run(inputs,targets)
# test
solver.test(inputs,targets)    


