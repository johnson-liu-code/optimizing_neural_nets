

import numpy as np


from genes.layer_type import layer_type
from genes.layer_type import layers_with_kernel
from genes.layer_type import layers_with_padding
from genes.layer_type import layers_with_activation
from genes.layer_type import layers_with_dimensionality
from genes.layer_type import layers_with_pooling
from genes.layer_type import layers_needing_strides
from genes.layer_type import layers_needing_strides
from genes.layer_type import layers_with_kernel_regularizer
from genes.layer_type import layers_with_kernel_constraint
from genes.layer_type import layers_with_bias_constraint
from genes.layer_type import layers_with_dilation_rate
from genes.layer_type import layers_with_groups
from genes.layer_type import layers_with_depth_multiplier
from genes.layer_type import layers_with_depthwise_initializer
from genes.layer_type import layers_with_depthwise_constraint


from genes.activation_type import activation_type
from genes.use_bias import use_bias
from genes.bias_initializer_type import bias_initializer_type
from genes.regularizer_type import regularizer_type
from genes.kernel_initializer_type import kernel_initializer_type

### Turn layer (a vector of and integers and floats) into strings
###     that are fed into the file that runs the neural network.

def get_phenotype(layer):
    first_layer_added = False
    first_layer_input_shape = False

    phenotype = "model.add("
    phenotype += layer_type[ layer.layer_type ] + "("

    #if layer[1] in layers_with_dimensionality:
    if layer.layer_type in layers_with_dimensionality:
        if first_layer_added == False:
            first_layer_added = True
            phenotype += "filters = " + str( layer.output_dimensionality )
        else:
            phenotype += ", filters = " + str( layer.output_dimensionality )

    #if layer[1] in layers_with_kernel:
    if layer.layer_type in layers_with_kernel:
        if first_layer_added == False:
            first_layer_added = True
            phenotype += "kernel_size = (" + str( layer.kernel_x ) + ", " + str( layer.kernel_y ) + ")"
        else:
            phenotype += ", kernel_size = (" + str( layer.kernel_x ) + ", " + str( layer.kernel_y ) + ")"

    #if layer[1] in layers_needing_strides:
    if layer.layer_type in layers_needing_strides:
        # If strides is set to None, False, or -1 in the layer vector, set strides to be equal to pool size.
        if layer.strides_x == None or layer.strides_x == False or layer.strides_x == -1:
            strides_x = layer.pool_size_x
        else:
            strides_x = layer.strides_x

        if layer.strides_y == None or layer.strides_y == False or layer.strides_y == -1:
            strides_y = layer.pool_size_y
        else:
            strides_y = layer.strides_y

        if first_layer_added == False:
            first_layer_added = True
            phenotype += "strdies = (" + str(strides_x) + ", " + str(strides_y) + ")"
        else:
            phenotype += ", strides = (" + str(strides_x) + ", " + str(strides_y) + ")"

    #if layer[1] in layers_with_activation:
    if layer.layer_type in layers_with_activation:
        phenotype += ", activation = " + activation_type[ layer.act ]
        phenotype += ", use_bias = " + use_bias[ layer.use_bias ]
        phenotype += ", bias_initializer = " + bias_initializer_type[ layer.bias_init ]
        phenotype += ", " + regularizer_type( 0, layer.bias_reg )
        phenotype += ", " + regularizer_type( 1, layer.act_reg )
        phenotype += ", kernel_initializer = " + kernel_initializer_type[ layer.kernel_init ]

    #if layer.layer_type in layers_with_kernel_regularizer:
    #    phenotype += ", " + regularizer_type( 2, layer.kernel_reg )

    #if layer.layer_type in layers_with_kernel_constraint:
    #    phenotype += ", kernel_constraint = " + kernel_constr

    #if layer[1] in layers_with_pooling:
    if layer.layer_type in layers_with_pooling:
        phenotype += ", pool_size = (" + str( layer.pool_size_x ) + ", " + str( layer.pool_size_y ) + ")"

    #if layer[1] in layers_with_padding:
    if layer.layer_type in layers_with_padding:
        if layer.padding == 0:
            phenotype += ", padding='same'"
        elif layer.padding == 1:
            phenotype += ", padding='valid'"

    if first_layer_input_shape == False:
        phenotype = phenotype + ', input_shape=x_train.shape[1:]))'
        first_layer_input_shape = True
    else:
        phenotype = phenotype + "))"

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
#  11: bias constraint
#  12: activation regularizer			number between 0 and 1
#  13: kernel initializer			0:10
#  14: kernel regularizer
#  15: kernel constraint
#  16: dilation rate
#  17: groups
#  18: depth_multiplier
#  19: depthwise_initializer
#  20: depthwise_regularizer
#  21: depthwise_constrint
#  22: dropout rate				number between 0 and 1
#  23: pool length x				1:10
#  24: pool length y				1:10
#  25: padding					0 ('same') or 1 ('valid')
