
'''
class regularizers:
    def __init__(self, num):
        reg = "regularizer=regularizers.l1(" + str(num) + ")"
        self.bias = "bias_" + reg
        self.activation = "activation_" + reg
'''
'''
import numpy as np
regularizers = [ "bias_regularizer=regularizers.l1({0})".format(x) for x in list(np.arange(0, 1, 0.01)) ]
#print list(np.arange(0, 1, 0.01))
#print regularizers[0]
#print regularizers[1]
'''

reg_types = ["bias_", "activity_"]

def regularizer_type(reg_type, num):
    reg = "regularizer=regularizers.l1(" + str(num) + ")"
    reg = reg_types[reg_type] + reg
    return reg
