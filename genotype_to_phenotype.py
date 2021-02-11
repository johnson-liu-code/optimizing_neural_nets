

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
from genes.layer_type import layers_with_dimensionality_input
from genes.layer_type import layers_with_pooling
from genes.layer_type import layers_needing_strides


from genes.activation_type import activation_type
from genes.use_bias import use_bias
from genes.bias_initializer_type import bias_initializer_type
from genes.regularizer_type import regularizer_type
from genes.kernel_initializer_type import kernel_initializer_type

### Turn layer (a vector of and integers and floats) into strings
###     that are fed into the file that runs the neural network.

def get_phenotype(layer):
    #print('layer: ', layer)
    #print('layer[0]: ', layer[0])

    first_layer_added = False
    first_layer_input_shape = False

    #print(layer[0])

    phenotype = "model.add("
    #phenotype += layer_type[layer[1]] + "("      # layer type
    phenotype += layer_type[ layer.layer_type ]

    ### This part is not needed.
    if layer.layer_type[1] in layers_with_dimensionality_input:
        if first_layer_added == False:
            first_layer_added = True
            phenotype += "input_dim=" + str(layer.layer_type)
        else:
            phenotype += ", input_dim=" + str(layer.layer_type)

    #if layer[1] in layers_with_dimensionality:
    if layer.layer_type in layers_with_dimensionality:
        if first_layer_added == False:
            first_layer_added = True
            #phenotype += "filters=" + str(layer[2])           # output_dimensionality
            phenotype += "filters = " + str( layer.output_dimensionality )
        else:
            #phenotype += ", " + str(layer[2])
            phenotype

    if layer.layer_type in layers_with_layer:
        if first_layer_added == False:
            first_layer_added = True
            phenotype += "LSTM(" + str(layer.layer_type) + ")"              #  18: layer
        else:
            phenotype += ", LSTM(" + str(layer.layer_type) + ")"

    if layer.layer_type in layers_with_maxlen:
        if first_layer_added == False:
            first_layer_added = True
            phenotype += str(chromosome[19])         #  19: maxlen
        else:
            phenotype += ", " + str(chromosome[19])

    if layer.layer_type in layers_with_vocab_size:
        if first_layer_added == False:
            first_layer_added = True
            phenotype += str(chromosome[20])         #  20: vocab_size
        else:
            phenotype += ", " + str(chromosome[20])

    if layer.layer_type in layers_with_dimensionality_embed:
        if first_layer_added == False:
            first_layer_added = True
            phenotype += str(chromosome[21])         #  21: Embedding size for each token
        else:
            phenotype += ", " + str(chromosome[21])

    if layer.layer_type in layers_with_num_heads:
        if first_layer_added == False:
            first_layer_added = True
            phenotype += str(chromosome[22])         #  22: Number of attention heads
        else:
            phenotype += ", " + str(chromosome[22])

    if layer.layer_type in layers_with_dimensionality_ff:
        if first_layer_added == False:
            first_layer_added = True
            phenotype += str(chromosome[23])  # 23: Hidden layer size in feed forward network inside transformer
        else:
            phenotype += ", " + str(chromosome[23])

    #if layer[1] in layers_with_kernel:
    if layer.layer_type in layers_with_kernel:
        if first_layer_added == False:
            first_layer_added = True
            #phenotype += "kernel_size=(" + str(layer[3]) + ", " + str(layer[4]) + ")" # kernel x size and kernel y size
            phenotype += "kernel_size = (" + str( layer.kernel_x ) + ", " + str( layer.kernel_y ) + ")"
        else:
            #phenotype += ", kernel_size=(" + str(layer[3]) + ", " + str(layer[4]) + ")" # kernel x size and kernel y size
            phenotype += ", kernel_size = (" + str( layer.kernel_x ) + ", " + str( layer.kernel_y ) + ")"

    #if layer[1] in layers_needing_strides:
    if layer.layer_type in layers_needing_strides:
        # If strides is set to None, False, or -1 in the layer vector, set strides to be equal to pool size.
        #if layer[5] == None or layer[5] == False or layer[5] == -1:
        if layer.strides_x == None or layer.strides_x == False or layer.strides_x == -1:
            #strides_x = layer[14]
            strides_x = layer.pool_size_x
        else:
            #strides_x = layer[5]
            strides_x = layer.strides_x

        #if layer[6] == None or layer[6] == False or layer[6] == -1:
        if layer.strides_y == None or layer.strides_y == False or layer.strides_y == -1:
            #strides_y = layer[15]
            strides_y = layer.pool_size_y
        else:
            #strides_y = layer[6]
            strides_y = layer.strides_y

        if first_layer_added == False:
            first_layer_added = True
            #phenotype += "strides=(" + str(strides_x) + "," + str(strides_y) + ")"
            phenotype += "strdies = (" + str(strides_x) + ", " + str(strides_y) + ")"
        else:
            #phenotype += ", strides=(" + str(strides_x) + "," + str(strides_y) + ")"
            phenotype += ", strides = (" + str(strides_x) + ", " + str(strides_y) + ")"

    '''
    if layer[1] in layers_with_pooling:
        if first_layer_added == False:
            first_layer_added = True
            phenotype += "pool_size = (1, 1)"
        else:
            phenotype += ", pool_size = (1, 1)"

    if layer[1] in layers_with_padding:
        if first_layer_added == False:
            first_layer_added = True
            phenotype += " padding='same'"
        else:
            phenotype += ", padding='same'"
    '''

    #if layer[1] in layers_with_activation:
    if layer.layer_type in layers_with_activation:
        #phenotype += ", activation=" + activation_type[layer[7]]          # activation type
        phenotype += ", activation = " + activation_type[ layer.act ]
        #phenotype += ", use_bias=" + use_bias[layer[8]]                 # use bias
        phenotype += ", use_bias = " + use_bias[ layer.use_bias ]
#        if layer[9] == -1 and layer[1] == 1:
#            
        #phenotype += ", bias_initializer=" + bias_initializer_type[layer[9]]    # bias initializer
        phenotype += ", bias_initializer = " + bias_initializer_type[ layer.bias_init ]
        #phenotype += ", " + regularizer_type(0, layer[10])      # bias regularizer
        phenotype += ", " + regularizer_type( 0, layer.bias_reg )
        #phenotype += ", " + regularizer_type(1, layer[11])      # activation regularizer
        phenotype += ", " + regularizer_type( 1, layer.act_reg )
        #phenotype += ", kernel_initializer=" + kernel_initializer_type[layer[12]]                  # kernel initializer
        phenotype += ", kernel_initializer = " + kernel_initializer_type[ layer.kernel_init ]

    #if layer[1] in layers_with_pooling:
    if layer.layer_type in layers_with_pooling:
        #phenotype += ", pool_size=(" + str(layer[14]) + ", " + str(layer[15]) + ")"
        phenotype += ", pool_size = (" + str( layer.pool_size_x ) + ", " + str( layer.pool_size_y ) + ")"

    #if layer[1] in layers_with_padding:
    if layer.layer_type in layers_with_padding:
        #if layer[16] == 0:
        if layer.padding == 0:
            phenotype += ", padding='same'"
        #elif layer[16] == 1:
        elif layer.padding == 1:
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
#   3: kernel x ratio                           In the range 0 to 1
#   4: kernel y ratio                           In the range 0 to 1
#   5: strides x length
#   6: strides y length
#   7: activation type				0:11
#   8: use bias?				0 (False) or 1 (True)
#   9: bias initializer				0:10
#  10: bias regularizer				number between 0 and 1
#  11: activation regularizer			number between 0 and 1
#  12: kernel initializer			0:10
#  13: dropout rate				number between 0 and 1
#  14: pool length x				1:10
#  15: pool length y				1:10
#  16: padding					0 ('same') or 1 ('valid')

'''
layer = np.random.randint(4)
activation = np.random.randint(11)
use_bias_type = np.random.randint(2)
bias_initializer = np.random.randint(16)
bias_regularizer = np.random.uniform()
activation_regularizer = np.random.uniform()
layer_02 = [layer, activation, use_bias_type, bias_initializer, bias_regularizer, activation_regularizer]
print(layer_02)
print(get_phenotype(layer_02))
'''
