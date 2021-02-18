
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

# https://keras.io/layers/core/

reg_types = ["bias_", "activity_", "kernel_", "depthwise_"]

l1l2 = ["l1", "l2", "l1_l2"]

def regularizer_type(reg_type, num, l1l2_type):
    if num == -1:
        reg = "regularizer=None"
    else:
        reg = "regularizer=regularizers." + l1l2[ l1l2_type ] + "(" + str(num) + ")"
    reg = reg_types[reg_type] + reg
    return reg
