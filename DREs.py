import math
import numpy as np
import pickle
import DRE_WM
import LSWT

#Data prediction
def eval(data,G=None):

    D = DRE_WM.DREs(data,G)

    #D = LSWT.DREs(data,G)
   
    return D