

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
from genes.kernel_initializer_type import kernel_initializer_type

### Turn chromosome (a vector of and integers and floats) into strings
### that are fed into the file that runs the neural network.

def get_phenotype(chromosome):
    #print('chromosome: ', chromosome)
    #print('chromosome[0]: ', chromosome[0])

    first_layer_added = False
    first_layer_input_shape = False

    #print(chromosome[0])

    phenotype = "model.add("
    phenotype += layer_type[chromosome[1]] + "("      # layer type

    ### This part is not needed.
    if chromosome[1] in layers_with_dimensionality:
        if first_layer_added == False:
            first_layer_added = True
            phenotype += "filters=" + str(chromosome[2])           # output_dimensionality
        else:
            phenotype += ", " + str(chromosome[2])

    if chromosome[1] in layers_with_kernel:
        if first_layer_added == False:
            first_layer_added = True
            phenotype += "kernel_size=(" + str(chromosome[3]) + ", " + str(chromosome[4]) + ")" # kernel x size and kernel y size
        else:
            phenotype += ", kernel_size=(" + str(chromosome[3]) + ", " + str(chromosome[4]) + ")" # kernel x size and kernel y size

    if chromosome[1] in layers_needing_strides:
        # If strides is set to None, False, or -1 in the chromosome vector, set strides to be equal to pool size.
        if chromosome[5] == None or chromosome[5] == False or chromosome[5] == -1:
            strides_x = chromosome[14]
        else:
            strides_x = chromosome[5]

        if chromosome[6] == None or chromosome[6] == False or chromosome[6] == -1:
            strides_y = chromosome[15]
        else:
            strides_y = chromosome[6]

        if first_layer_added == False:
            first_layer_added = True
            phenotype += "strides=(" + str(strides_x) + "," + str(strides_y) + ")"
        else:
            phenotype += ", strides=(" + str(strides_x) + "," + str(strides_y) + ")"

    '''
    if chromosome[1] in layers_with_pooling:
        if first_layer_added == False:
            first_layer_added = True
            phenotype += "pool_size = (1, 1)"
        else:
            phenotype += ", pool_size = (1, 1)"

    if chromosome[1] in layers_with_padding:
        if first_layer_added == False:
            first_layer_added = True
            phenotype += " padding='same'"
        else:
            phenotype += ", padding='same'"
    '''

    if chromosome[1] in layers_with_activation:
        phenotype += ", activation=" + activation_type[chromosome[7]]          # activation type
        phenotype += ", use_bias=" + use_bias[chromosome[8]]                 # use bias
#        if chromosome[9] == -1 and chromosome[1] == 1:
#            
        phenotype += ", bias_initializer=" + bias_initializer_type[chromosome[9]]    # bias initializer
        phenotype += ", " + regularizer_type(0, chromosome[10])      # bias regularizer
        phenotype += ", " + regularizer_type(1, chromosome[11])      # activation regularizer
        phenotype += ", kernel_initializer=" + kernel_initializer_type[chromosome[12]]                  # kernel initializer

    if chromosome[1] in layers_with_pooling:
        phenotype += ", pool_size=(" + str(chromosome[14]) + ", " + str(chromosome[15]) + ")"

    if chromosome[1] in layers_with_padding:
        if chromosome[16] == 0:
            phenotype += ", padding='same'"
        elif chromosome[16] == 1:
            phenotype += ", padding='valid'"


    if first_layer_input_shape == False:
        phenotype = phenotype + ', input_shape=x_train.shape[1:]))'
        first_layer_input_shape = True
    else:
        phenotype = phenotype + "))"
    #print(phenotype)
    return phenotype

#   0: expression on/off                        0 (skip layer), 1 (use layer), 2 (dropout layer), 3 (flatten layer)
#   1: layer type				0:5
#   2: output dimensionality (called filters by Keras)
#   3: kernel x
#   4: kernel y
#   5: strides x
#   6: strides y
#   7: activation type				0:11
#   8: use bias?				0 (False) or 1 (True)
#   9: bias initializer				0:10
#  10: bias regularizer				number between 0 and 1
#  11: activation regularizer			number between 0 and 1
#  12: kernel initializer			0:10
#  13: dropout rate				number between 0 and 1
#  14: pool size x				1:10
#  15: pool size y				1:10
#  16: padding					0 ('same') or 1 ('valid')

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
