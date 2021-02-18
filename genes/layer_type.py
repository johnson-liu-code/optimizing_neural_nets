# Type of layer.
# https://keras.io/layers/about-keras-layers/

layer_type = {  0: "Dense",
                1: "Conv2D",
                2: "SeparableConv2D",
                3: "DepthwiseConv2D",
                4: "MaxPooling2D",
                5: "AveragePooling2D"  }

layers_with_padding = [1, 2, 3, 4, 5]
layers_with_kernel = [1, 2, 3]
layers_with_activation = [0, 1, 2, 3]
layers_with_dimensionality = [0, 1, 2]
layers_with_pooling = [4, 5]
layers_needing_strides = [1, 2, 3, 4, 5]
layers_with_kernel_regularizer = [0, 1]
#layers_with_kernel_constraint = [0, 1]
#layers_with_bias_constraint = [0, 1, 2, 3]
layers_with_dilation_rate = [1, 2, 3]
#layers_with_groups = [1]
layers_with_depth_multiplier = [2, 3]
layers_with_depthwise_initializer = [2, 3]
#layers_with_pointwise_initializer = [2]
layers_with_depthwise_constraint = [2, 3]
#layers_with_pointwise_constraint = [2]

# keras.layers.Dense(units, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
# keras.layers.Conv2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
# keras.layers.SeparableConv2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), depth_multiplier=1, activation=None, use_bias=True, depthwise_initializer='glorot_uniform', pointwise_initializer='glorot_uniform', bias_initializer='zeros', depthwise_regularizer=None, pointwise_regularizer=None, bias_regularizer=None, activity_regularizer=None, depthwise_constraint=None, pointwise_constraint=None, bias_constraint=None)
# keras.layers.DepthwiseConv2D(kernel_size, strides=(1, 1), padding='valid', depth_multiplier=1, data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, depthwise_initializer='glorot_uniform', bias_initializer='zeros', depthwise_regularizer=None, bias_regularizer=None, activity_regularizer=None, depthwise_constraint=None, bias_constraint=None)
# keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)
# keras.layers.AveragePooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)
# keras.layers.Dropout(.2)
