import math


from genes.activation_type import activation_type
from genes.use_bias import use_bias
from genes.bias_initializer_type import bias_initializer_type
from genes.regularizer_type import regularizer_type
from genes.kernel_initializer_type import kernel_initializer_type


def get_phenotype( layer, first_expressed_layer_added, x_dimension, y_dimension ):
    ########### Dense layer. ######################################################################################################
    if layer.layer_type == 0:
        phenotype  = 'model.add( Dense( units = {}, '.format( layer.output_dimensionality )
        phenotype += 'activation = {}, '.format( activation_type[ layer.act ] )
        phenotype += 'use_bias = {}, '.format( use_bias[ layer.use_bias ] )
        phenotype += 'bias_initializer = {}, '.format( bias_initializer_type[ layer.bias_init ] )
        phenotype += '{}, '.format( regularizer_type( 0, layer.bias_reg, layer.bias_reg_l1l2_type ) )
        phenotype += '{}, '.format( regularizer_type( 1, layer.act_reg, layer.act_reg_l1l2_type ) )
        phenotype += 'kernel_initializer = {}, '.format( kernel_initializer_type[ layer.kernel_init ] )
        phenotype += '{}'.format( regularizer_type( 2, layer.kernel_reg, layer.kernel_reg_l1l2_type ) )

    ###############################################################################################################################


    ########## Conv2D layer. ######################################################################################################
    elif layer.layer_type == 1:
        phenotype  = 'model.add( Conv2D( filters = {}, '.format( layer.output_dimensionality )

        kernel_x = max( 1, math.floor( layer.kernel_x_ratio * x_dimension ) )
        kernel_y = max( 1, math.floor( layer.kernel_y_ratio * y_dimension ) )

        phenotype += 'kernel_size = ( {}, {} ), '.format( str(kernel_x), str(kernel_y) )

        stride_x = max( 1, math.floor( layer.stride_x_ratio * x_dimension ) )

        if layer.layer_type == 2 or layer.layer_type == 3:
            stride_y = stride_x
        else:
            stride_y = max( 1, math.floor( layer.stride_y_ratio * y_dimension ) )

        phenotype += 'strides = ( {}, {} ), '.format( str(stride_x), str(stride_y) )

        phenotype += 'activation = {}, '.format( activation_type[ layer.act ] )
        phenotype += 'use_bias = {}, '.format( use_bias[ layer.use_bias ] )
        phenotype += 'bias_initializer = {}, '.format( bias_initializer_type[ layer.bias_init ] )
        phenotype += '{}, '.format( regularizer_type( 0, layer.bias_reg, layer.bias_reg_l1l2_type ) )
        phenotype += '{}, '.format( regularizer_type( 1, layer.act_reg, layer.act_reg_l1l2_type ) )
        phenotype += 'kernel_initializer = {}, '.format( kernel_initializer_type[ layer.kernel_init ] )
        phenotype += '{}, '.format( regularizer_type( 2, layer.kernel_reg, layer.kernel_reg_l1l2_type ) )

        if layer.padding == 0:
            phenotype += "padding = 'same'"
        elif layer.padding == 1:
            phenotype += "padding = 'valid'"

    ###############################################################################################################################


    ########## SeparableConv2D ####################################################################################################
    elif layer.layer_type == 2:
        phenotype  = 'model.add( SeparableConv2D( filters = {}, '.format( layer.output_dimensionality )

        kernel_x = max( 1, math.floor( layer.kernel_x_ratio * x_dimension ) )
        kernel_y = max( 1, math.floor( layer.kernel_y_ratio * y_dimension ) )

        phenotype += 'kernel_size = ( {}, {} ), '.format( str(kernel_x), str(kernel_y) )

        stride_x = max( 1, math.floor( layer.stride_x_ratio * x_dimension ) )

        if layer.layer_type == 2 or layer.layer_type == 3:
            stride_y = stride_x
        else:
            stride_y = max( 1, math.floor( layer.stride_y_ratio * y_dimension ) )

        phenotype += 'strides = ( {}, {} ), '.format( str(stride_x), str(stride_y) )
        phenotype += 'activation = {}, '.format( activation_type[ layer.act ] )
        phenotype += 'use_bias = {}, '.format( use_bias[ layer.use_bias ] )
        phenotype += 'bias_initializer = {}, '.format( bias_initializer_type[ layer.bias_init ] )
        phenotype += '{}, '.format( regularizer_type( 0, layer.bias_reg, layer.bias_reg_l1l2_type ) )
        phenotype += '{}, '.format( regularizer_type( 1, layer.act_reg, layer.act_reg_l1l2_type ) )
        phenotype += 'kernel_initializer = {}, '.format( kernel_initializer_type[ layer.kernel_init ] )
        phenotype += '{}, '.format( regularizer_type( 2, layer.kernel_reg, layer.kernel_reg_l1l2_type ) )

        if layer.padding == 0:
            phenotype += "padding = 'same'"
        elif layer.padding == 1:
            phenotype += "padding = 'valid'"

    ###############################################################################################################################


    ########## DepthwiseConv2D ####################################################################################################
    elif layer.layer_type == 3:
        kernel_x = max( 1, math.floor( layer.kernel_x_ratio * x_dimension ) )
        kernel_y = max( 1, math.floor( layer.kernel_y_ratio * y_dimension ) )

        phenotype  = 'model.add( DepthwiseConv2D( kernel_size = ( {}, {} ), '.format( str(kernel_x), str(kernel_y) )

        stride_x = max( 1, math.floor( layer.stride_x_ratio * x_dimension ) )

        if layer.layer_type == 2 or layer.layer_type == 3:
            stride_y = stride_x
        else:
            stride_y = max( 1, math.floor( layer.stride_y_ratio * y_dimension ) )

        phenotype += 'strides = ( {}, {} ), '.format( str(stride_x), str(stride_y) )
        phenotype += 'activation = {}, '.format( activation_type[ layer.act ] )
        phenotype += 'use_bias = {}, '.format( use_bias[ layer.use_bias ] )
        phenotype += 'bias_initializer = {}, '.format( bias_initializer_type[ layer.bias_init ] )
        phenotype += '{}, '.format( regularizer_type( 0, layer.bias_reg, layer.bias_reg_l1l2_type ) )
        phenotype += '{}, '.format( regularizer_type( 1, layer.act_reg, layer.act_reg_l1l2_type ) )
        phenotype += 'kernel_initializer = {}, '.format( kernel_initializer_type[ layer.kernel_init ] )
        phenotype += '{}, '.format( regularizer_type( 2, layer.kernel_reg, layer.kernel_reg_l1l2_type ) )

        if layer.padding == 0:
            phenotype += "padding = 'same'"
        elif layer.padding == 1:
            phenotype += "padding = 'valid'"

    ###############################################################################################################################


    ########## MaxPooling2D ####################################################################################################
    elif layer.layer_type == 4:
        # If strides is set to None, False, or -1 in the layer vector, set strides to be equal to pool size.
        if layer.stride_x_ratio == None or layer.stride_x_ratio == False or layer.stride_x_ratio == -1:
            pool_x = max( 1, math.floor( layer.pool_x_ratio * x_dimension ) )
            stride_x = pool_x
        else:
            stride_x = max( 1, math.floor( layer.stride_x_ratio * x_dimension ) )

        if layer.layer_type == 2 or layer.layer_type == 3:
            stride_y = stride_x
        else:
            if layer.stride_y_ratio == None or layer.stride_y_ratio == False or layer.stride_y_ratio == -1:
                pool_y = max( 1, math.floor( layer.pool_y_ratio * y_dimension ) )
                stride_y = pool_y
            else:
                stride_y = max( 1, math.floor( layer.stride_y_ratio * y_dimension ) )

        phenotype  = 'model.add( MaxPooling2D( strides = ( {}, {} ), '.format( str(stride_y), str(stride_x) )

        pool_x = max( 1, math.floor( layer.pool_x_ratio * x_dimension ) )
        pool_y = max( 1, math.floor( layer.pool_y_ratio * y_dimension ) )

        phenotype += 'pool_size = ( {}, {} ), '.format( str(pool_x), str(pool_y) )

        if layer.padding == 0:
            phenotype += "padding = 'same'"
        elif layer.padding == 1:
            phenotype += "padding = 'valid'"

    ###############################################################################################################################


    ########## AveragePooling2D ####################################################################################################
    elif layer.layer_type == 5:
        # If strides is set to None, False, or -1 in the layer vector, set strides to be equal to pool size.
        if layer.stride_x_ratio == None or layer.stride_x_ratio == False or layer.stride_x_ratio == -1:
            pool_x = max( 1, math.floor( layer.pool_x_ratio * x_dimension ) )
            stride_x = pool_x
        else:
            stride_x = max( 1, math.floor( layer.stride_x_ratio * x_dimension ) )

        if layer.layer_type == 2 or layer.layer_type == 3:
            stride_y = stride_x
        else:
            if layer.stride_y_ratio == None or layer.stride_y_ratio == False or layer.stride_y_ratio == -1:
                pool_y = max( 1, math.floor( layer.pool_y_ratio * y_dimension ) )
                stride_y = pool_y
            else:
                stride_y = max( 1, math.floor( layer.stride_y_ratio * y_dimension ) )

        phenotype  = 'model.add( AveragePooling2D( strides = ( {}, {} ), '.format( str(stride_y), str(stride_x) )

        pool_x = max( 1, math.floor( layer.pool_x_ratio * x_dimension ) )
        pool_y = max( 1, math.floor( layer.pool_y_ratio * y_dimension ) )

        phenotype += 'pool_size = ( {}, {} ), '.format( str(pool_x), str(pool_y) )

        if layer.padding == 0:
            phenotype += "padding = 'same'"
        elif layer.padding == 1:
            phenotype += "padding = 'valid'"

    ###############################################################################################################################


    if first_expressed_layer_added == False:
        phenotype += ', input_shape=x_train.shape[1:] ) )'
    else:
        phenotype += ' ) )'

    return phenotype
