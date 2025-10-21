import math
import numpy as np
import pickle
import DREs

with open("wind_forces_inputs.pkl", "rb") as f: 
    data, G = pickle.load(f)

D = DREs.eval(data,G)


print(D[2]['CL'])
