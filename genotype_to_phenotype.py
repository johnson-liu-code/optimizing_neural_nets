

import numpy as np


#from layer_type import layer_type
#from activation_type import activation_type
#from use_bias import use_bias
#from bias_initializer_type import bias_initializer_type
#from regularizers import regularizer_type

from genes.layer_type import layer_type
from genes.layer_type import layers_with_kernel
from genes.layer_type import layers_with_padding
from genes.layer_type import layers_with_activation
from genes.layer_type import layers_with_dimensionality
from genes.layer_type import layers_with_pooling
from genes.layer_type import layers_needing_strides

from genes.activation_type import activation_type
from genes.use_bias import use_bias
from genes.bias_initializer_type import bias_initializer_type
from genes.regularizer_type import regularizer_type

### Turn chromosome (a vector of and integers and floats) into strings
### that are fed into the file that runs the neural network.

def get_phenotype(chromosome):
    #print('chromosome: ', chromosome)
    #print('chromosome[0]: ', chromosome[0])

    first_parameter = False

    #print(chromosome[0])

    phenotype = "model.add("
    phenotype += layer_type[chromosome[0]] + "("                    # layer type

    if chromosome[0] in layers_with_dimensionality:
        if first_parameter == False:
            first_parameter = True
            phenotype += str(chromosome[1])
        else:
            phenotype += ", " + str(chromosome[1])                             # number of output_dimensionality

    if chromosome[0] in layers_with_kernel:
        if first_parameter == False:
            first_parameter = True
            phenotype += "(" + str(chromosome[2]) + ", " + str(chromosome[3]) + ")" # kernel x size and kernel y size
        else:
            phenotype += ", (" + str(chromosome[2]) + ", " + str(chromosome[3]) + ")" # kernel x size and kernel y size

    if chromosome[0] in layers_needing_strides:
        if first_parameter == False:
            first_parameter = True
            phenotype += "strides=(" + str(chromosome[5]) + "," + str(chromosome[5]) + ") "
        else:
            phenotype += ", strides=(" + str(chromosome[5]) + "," + str(chromosome[5]) + ") "

    if chromosome[0] in layers_with_pooling:
        if first_parameter == False:
            first_parameter = True
            phenotype += "pool_size = (1, 1)"
        else:
            phenotype += ", pool_size = (1, 1)"

    if chromosome[0] in layers_with_padding:
        if first_parameter == False:
            first_parameter = True
            phenotype += " padding='same'"
        else:
            phenotype += ", padding='same'"

    if chromosome[0] in layers_with_activation:
        phenotype += ", " + activation_type[chromosome[4]]          # activation type
        phenotype += ", " + use_bias[chromosome[6]]                 # use bias
        phenotype += ", " + bias_initializer_type[chromosome[7]]    # bias initializer
        phenotype += ", " + regularizer_type(0, chromosome[8])      # bias regularizer
        phenotype += ", " + regularizer_type(1, chromosome[9])      # activation regularizer

    phenotype = phenotype + "))"
    #print(phenotype)
    return phenotype

#  0: layer type				0:5
#  1: number of output_dimensionality
#  2: kernel x
#  3: kernel y
#  4: activation type				0:11
#  5: use bias?					0 or 1
#  6: bias initializer				0:10
#  7: bias regularizer				random number between 0 and 1
#  8: activation regularizer			random number between 0 and 1
#  9: x_strides
# 10: y_strides

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
