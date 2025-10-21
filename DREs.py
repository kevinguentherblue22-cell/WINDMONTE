import math
import numpy as np
import pickle
import DRE_WM

#Data prediction
def eval(data,G=None):

    D = DRE_WM.DREs(data,G)
   
    return D