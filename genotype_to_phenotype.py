

import numpy as np


from genes.layer_type import layer_type
from genes.layer_type import layers_with_kernel
from genes.layer_type import layers_with_padding
from genes.layer_type import layers_with_activation
from genes.layer_type import layers_with_dimensionality
from genes.layer_type import layers_with_dimensionality_input
from genes.layer_type import layers_with_pooling
from genes.layer_type import layers_needing_strides
from genes.layer_type import layers_with_kernel_regularizer
#from genes.layer_type import layers_with_kernel_constraint
#from genes.layer_type import layers_with_bias_constraint
#from genes.layer_type import layers_with_dilation_rate
#from genes.layer_type import layers_with_groups
#from genes.layer_type import layers_with_depth_multiplier
#from genes.layer_type import layers_with_depthwise_initializer
#from genes.layer_type import layers_with_depthwise_constraint


from genes.activation_type import activation_type
from genes.use_bias import use_bias
from genes.bias_initializer_type import bias_initializer_type
from genes.regularizer_type import regularizer_type
from genes.kernel_initializer_type import kernel_initializer_type


### Turn layer (a vector of and integers and floats) into strings
###     that are fed into the file that runs the neural network.

def get_phenotype( layer, first_expressed_layer_added, x_dimension, y_dimension ):
    phenotype_string_list = []
    #first_argument_added = False
    first_layer_input_shape = False

    phenotype = "model.add("
    phenotype += layer_type[ layer.layer_type ] + "("

    #if layer[1] in layers_with_dimensionality:

    if layer.layer_type==6:
        phenotype = "model.add(Embedding(" + str(layer.input_dimensionality) + "," + str(layer.output_dimensionality) +"))"
    if layer.layer_type==7:
        phenotype = "model.add(Bidirectional(LSTM(" + str( layer.layervalue ) + ")))"
    if layer.layer_type==8:
        phenotype = "model.add(GlobalAveragePooling1D())"
    if layer.layer_type==9:
        phenotype = "model.add(Conv1D(" + str( layer.filters) + "," + str( layer.kernel ) + "))"
    if layer.layer_type==10:
        phenotype = "model.add(MaxPooling1D(pool_size=" + str( layer.pool) + ", strides=" + str( layer.strides) + r", padding='same'" +"))"
    if layer.layer_type==11:
        phenotype = "model.add(GlobalMaxPooling1D())"
    if layer.layer_type==12:
        phenotype = "model.add(LSTM(" + str( layer.layervalue ) + "))"
    if layer.layer_type==13:
        phenotype = "model.add(LayerNormalization())"

    if layer.layer_type in layers_with_dimensionality:
        #if first_argument_added == False:
        #    first_argument_added = True
        if layer.layer_type == 0:
            phenotype_string_list.append( "units = " + str( layer.output_dimensionality ) )
        else:
            phenotype_string_list.append( "filters = " + str( layer.output_dimensionality ) )
        #else:
        #    if layer.layer_type == 0:
        #        phenotype += ", units = " + str( layer.output_dimensionality )
        #    else:
        #        phenotype += ", filters = " + str( layer.output_dimensionality )

    #if layer[1] in layers_with_kernel:
    if layer.layer_type in layers_with_kernel:
        #if first_argument_added == False:
        #    first_argument_added = True

        kernel_x = layer.kernel_x_ratio * x_dimension
        kernel_y = layer.kernel_y_ratio * y_dimension

        phenotype_string_list.append( "kernel_size = (" + str( kernel_x ) + ", " + str( kernel_y ) + ")" )
        #else:
        #    phenotype += ", kernel_size = (" + str( layer.kernel_x ) + ", " + str( layer.kernel_y ) + ")"

    #if layer[1] in layers_needing_strides:
    if layer.layer_type in layers_needing_strides:
        # If strides is set to None, False, or -1 in the layer vector, set strides to be equal to pool size.
        if layer.stride_x == None or layer.stride_x == False or layer.stride_x == -1:
            pool_x = layer.pool_x_ratio * x_dimension
            stride_x = pool_x
        else:
            stride_x = layer.stride_x_ratio * x_dimension

        if layer.stride_y == None or layer.stride_y == False or layer.stride_y == -1:
            pool_y = layer.pool_y_ratio * y_dimension
            stride_y = pool_y
        else:
            stride_y = layer.stride_y_ratio * y_dimension

        if layer.layer_type == 3:
            stride_y = stride_x

        #if first_argument_added == False:
        #    first_argument_added = True
        phenotype_string_list.apend( "strides = (" + str(stride_x) + ", " + str(stride_y) + ")" )
        #else:
        #    phenotype += ", strides = (" + str(stride_x) + ", " + str(stride_y) + ")"

    #if layer[1] in layers_with_activation:
    if layer.layer_type in layers_with_activation:
        phenotype_string_list.append( "activation = " + activation_type[ layer.act ] )
        phenotype_string_list.append( "use_bias = " + use_bias[ layer.use_bias ] )
        phenotype_string_list.append( "bias_initializer = " + bias_initializer_type[ layer.bias_init ] )
        phenotype_string_list.append( regularizer_type( 0, layer.bias_reg, layer.bias_reg_l1l2_type ) )
        phenotype_string_list.append( regularizer_type( 1, layer.act_reg, layer.act_reg_l1l2_type ) )
        phenotype += ", kernel_initializer = " + kernel_initializer_type[ layer.kernel_init ]

    if layer.layer_type in layers_with_kernel_regularizer:
        phenotype_string_list.append( regularizer_type( 2, layer.kernel_reg, layer.kernel_reg_l1l2_type ) )
    

    #if layer.layer_type in layers_with_kernel_regularizer:
    #    phenotype += ", " + regularizer_type( 2, layer.kernel_reg )

    #if layer.layer_type in layers_with_kernel_constraint:
    #    phenotype += ", kernel_constraint = " + kernel_constr

    #if layer[1] in layers_with_pooling:
    if layer.layer_type in layers_with_pooling:
        pool_x = layer.pool_x_ratio * x_dimension
        pool_y = layer.pool_y_ratio * y_dimension
        phenotype_string_list.append( "pool_size = (" + str( pool_x ) + ", " + str( layer.pool_y ) + ")" )

    #if layer[1] in layers_with_padding:
    if layer.layer_type in layers_with_padding:
        if layer.padding == 0:
            phenotype_string_list.append( "padding='same'" )
        elif layer.padding == 1:
            phenotype_string_list.append( "padding='valid'" )

    # This doesn't work. This layer does not know about the layers that came before it.
    #if first_layer_input_shape == False:
    #    phenotype = phenotype + ', input_shape=x_train.shape[1:]))'
    #    first_layer_input_shape = True
    #else:
    #    phenotype = phenotype + "))"

    if first_expressed_layer_added == False:
        phenotype_string_list.append( "input_shape=x_train.shape[1:]))" )
    else:
        phenotype_string_list.append( "))" )

    phenotype = phenotype + " ,".join( phenotype_string_list )


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
