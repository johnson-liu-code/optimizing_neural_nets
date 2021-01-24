# Type of layer.
# https://keras.io/layers/about-keras-layers/

# DepthwiseConv2D doesn't work for some reason

layer_type = {  0: "Dense",
                1: "Conv2D",
                2: "SeparableConv2D",
                3: "DepthwiseConv2D",
                4: "MaxPooling2D",
                5: "AveragePooling2D",
                6: "Embedding",
                7: "Bidirectional",
                8: "TokenAndPositionEmbedding",
                9: "TransformerBlock",
                10: "GlobalAveragePooling1D"
                }


layers_with_dimensionality_input = [6]
layers_with_dimensionality = [0, 1, 2, 6]
layers_with_layer = [7]
layers_with_maxlen = [8]
layers_with_vocab_size = [8]
layers_with_dimensionality_embed = [8, 9]
layers_with_num_heads = [9]
layers_with_dimensionality_ff = [9]
layers_with_kernel = [1, 2, 3]
layers_needing_strides = [1, 2, 3, 4, 5]
layers_with_pooling = [4, 5]
layers_with_padding = [1, 2, 3, 4, 5]
layers_with_activation = [0, 1, 2, 3]




# keras.layers.Dense(units, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
# keras.layers.Conv2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
# keras.layers.SeparableConv2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), depth_multiplier=1, activation=None, use_bias=True, depthwise_initializer='glorot_uniform', pointwise_initializer='glorot_uniform', bias_initializer='zeros', depthwise_regularizer=None, pointwise_regularizer=None, bias_regularizer=None, activity_regularizer=None, depthwise_constraint=None, pointwise_constraint=None, bias_constraint=None)
# keras.layers.DepthwiseConv2D(kernel_size, strides=(1, 1), padding='valid', depth_multiplier=1, data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, depthwise_initializer='glorot_uniform', bias_initializer='zeros', depthwise_regularizer=None, bias_regularizer=None, activity_regularizer=None, depthwise_constraint=None, bias_constraint=None)
# keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)
# keras.layers.AveragePooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)
# keras.layers.Dropout(.2)

# tf.keras.layers.Embedding(input_dim,output_dim,embeddings_initializer="uniform",embeddings_regularizer=None,activity_regularizer=None,embeddings_constraint=None,mask_zero=False,input_length=None,**kwargs)
# tf.keras.layers.Bidirectional(layer, merge_mode="concat", weights=None, backward_layer=None, **kwargs)
# TokenAndPositionEmbedding(self, maxlen, vocab_size, embed_dim)
# TransformerBlock(self, embed_dim, num_heads, ff_dim, rate=0.1)
# tf.keras.layers.GlobalAveragePooling1D(data_format="channels_last", **kwargs)
