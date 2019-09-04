

import numpy as np


from layer_type import layer_type
from activation_type import activation_type
from use_bias import use_bias
from bias_initializer_type import bias_initializer_type
from regularizers import regularizer_type


def get_phenotype(chromosome):
    #print('chromosome: ', chromosome)
    #print('chromosome[0]: ', chromosome[0])
    phenotype = "model.add("
    phenotype += layer_type[chromosome[0]] + "("                # layer type
    phenotype += str(chromosome[1]) + ", "
    if chromosome[0] != 0:
        phenotype += "(" + str(chromosome[2]) + ", " + str(chromosome[3]) + "), "
    phenotype += activation_type[chromosome[4]] + ", "          # activation type
    phenotype += use_bias[chromosome[5]] + ", "                 # use bais
    phenotype += bias_initializer_type[chromosome[6]] + ", "    # bias initializer
    phenotype += regularizer_type(0, chromosome[7]) + ", "    # bias regularizer
    phenotype += regularizer_type(1, chromosome[8])           # activation regularizer

    return phenotype + "))"

# 1: layer type					0:4
# 2: number of nodes
# 3: activation type				0:11
# 4: use bias?					True or False
# 5: bias initializer				0:10
# 6: bias regularizer				random number between 0 and 1
# 7: activation regularizer			random number between 0 and 1


'''
layer = np.random.randint(4)
activation = np.random.randint(11)
use_bias_type = np.random.randint(2)
bias_initializer = np.random.randint(16)
bias_regularizer = np.random.uniform()
activation_regularizer = np.random.uniform()
chromosome_02 = [layer, activation, use_bias_type, bias_initializer, bias_regularizer, activation_regularizer]
print(chromosome_02)
print(get_phenotype(chromosome_02))
'''
