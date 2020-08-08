import numpy as np
import pickle

folder = 'J_A+A_636_A24/'
basenames = ['il3n1', 'il7n3', 'il9n6', 'il11n7', 'il13n10', 'il15n11']
files = [folder + f for f in [f + '.dat' for f in basenames]]

quads = {k: np.genfromtxt(f) for k, f in zip(basenames, files)}

with open('Quadratures.pickle', 'wb') as pkl:
    pickle.dump(quads, pkl)