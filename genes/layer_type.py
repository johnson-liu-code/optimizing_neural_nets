# Type of layer.
# https://keras.io/layers/about-keras-layers/

layer_type = {  0: "Dense",
                1: "Conv2D",
                2: "SeparableConv2D",
                3: "DepthwiseConv2D",
                4: "MaxPooling2D",
                5: "AveragePooling2D",
                6: "LSTM",
                7: "Bidirectional",
                8: "Conv1D",
                9: "MaxPooling1D",
                10: "GlobalMaxPooling1D",
                11: "AveragePooling1D",
                12: "GlobalAveragePooling1D"
                }

layers_with_layer = [6] #LSTM
layers_with_filters = [8] #Conv1D",



layers_with_padding = [1, 2, 3, 4, 5]
layers_with_kernel = [1, 2, 3, 9]
layers_with_activation = [0, 1, 2, 3]
layers_with_dimensionality = [0, 1, 2, 6]
layers_with_pooling = [4, 5, 10]
layers_needing_strides = [1, 2, 3, 4, 5, 10]
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

# tf.keras.layers.Embedding(input_dim,output_dim,embeddings_initializer="uniform",embeddings_regularizer=None,activity_regularizer=None,embeddings_constraint=None,mask_zero=False,input_length=None,**kwargs)
# tf.keras.layers.Bidirectional(layer, merge_mode="concat", weights=None, backward_layer=None, **kwargs)
# TokenAndPositionEmbedding(self, maxlen, vocab_size, embed_dim)
# TransformerBlock(self, embed_dim, num_heads, ff_dim, rate=0.1)
# tf.keras.layers.GlobalAveragePooling1D(data_format="channels_last", **kwargs)
# tf.keras.layers.Conv1D(filters,kernel_size,strides=1,padding="valid",data_format="channels_last",dilation_rate=1,groups=1,activation=None,use_bias=True,kernel_initializer="glorot_uniform",bias_initializer="zeros",kernel_regularizer=None,bias_regularizer=None,activity_regularizer=None,kernel_constraint=None,bias_constraint=None,**kwargs)
# tf.keras.layers.MaxPooling1D(pool_size=2, strides=None, padding="valid", data_format="channels_last", **kwargs)
# tf.keras.layers.GlobalMaxPooling1D(data_format="channels_last", **kwargs)
# tf.keras.layers.LSTM(units,activation="tanh",recurrent_activation="sigmoid",use_bias=True,kernel_initializer="glorot_uniform",recurrent_initializer="orthogonal",bias_initializer="zeros",unit_forget_bias=True,kernel_regularizer=None,recurrent_regularizer=None,bias_regularizer=None,activity_regularizer=None,kernel_constraint=None,recurrent_constraint=None,bias_constraint=None,dropout=0.0,recurrent_dropout=0.0,return_sequences=False,return_state=False,go_backwards=False,stateful=False,time_major=False,unroll=False,**kwargs)
# tf.keras.layers.LayerNormalization(axis=-1,epsilon=0.001,center=True,scale=True,beta_initializer="zeros",gamma_initializer="ones",beta_regularizer=None,gamma_regularizer=None,beta_constraint=None,gamma_constraint=None,trainable=True,name=None,**kwargs)



