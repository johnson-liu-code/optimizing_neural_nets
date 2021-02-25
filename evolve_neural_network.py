import sys
import os
import pickle
import random
import numpy as np
import subprocess
import math
import time
import datetime
import itertools
from multiprocessing import Pool as ThreadPool

import deap
from deap import base
from deap import creator
from deap import tools
#from keras.datasets import reuters

from genotype_to_phenotype import get_phenotype
from functions.divide_chunks import divide_chunks
from genes.layer_type import layers_with_kernel
from genes.layer_type import layers_with_pooling


np.random.seed(0)


infile_name = sys.argv[1]

with open( infile_name, 'r' ) as fil:
    lines = fil.readlines()

### Number of layers beyond the first layer. The first layer is a
### special layer that cannot be a flatten or dropout layer.
max_num_layers = int( lines[0].split()[2] )
population_size = int( lines[1].split()[2] )

selection_size = int( lines[2].split()[2] )
migration_size = int( lines[3].split()[2] )

layer_expression_rate = float( lines[4].split()[2] )
mutation_probability = float( lines[5].split()[2] )

crossover_probability = float( lines[6].split()[2] )
number_of_generations = int( lines[7].split()[2] )

batch_size = int( lines[8].split()[2] )
number_of_classes = int( lines[9].split()[2] )

epochs = int( lines[10].split()[2] )
initial_population_directory = lines[11].split()[2]

### Top and bottom wrapper text to be used in making the neural network.
top_file = 'wrapper_text/top_text.txt'
bot_file = 'wrapper_text/bot_text.txt'

### Read in the lines from the top and bot text files.
with open( top_file, 'r' ) as top:
    top_lines = top.readlines()

with open( bot_file, 'r' ) as bot:
    bot_lines = bot.readlines()

### Get today's date. Used in naming data directories.
today = datetime.datetime.now()
year = today.year
month = today.month
day = today.day

### Name the data directories.
data_directory_name = 'data/{0}/{0}{1:02d}/{0}{1:02d}{2:02d}/'.format( year, month, day )
neural_net_directory_name = 'neural_network_files/{0}/{0}{1:02d}/{0}{1:02d}{2:02d}/'.format( year, month, day )
output_directory_name = 'output_files/{0}/{0}{1:02d}/{0}{1:02d}{2:02d}/'.format( year, month, day )

### Create data directories.
if not os.path.isdir( data_directory_name ):
    os.makedirs( data_directory_name )

if not os.path.isdir( neural_net_directory_name ):
    os.makedirs( neural_net_directory_name )

if not os.path.isdir( output_directory_name ):
    os.makedirs( output_directory_name )

### Determine the next available directory ID number for today.
if not os.listdir( data_directory_name ):
    next_dir_number = 1

else:
    last_dir_name = sorted( os.listdir( data_directory_name ) )
    last_dir_number = int( last_dir_name[-1] )
    next_dir_number = last_dir_number + 1

data_dir_number_name = data_directory_name + '{0:04d}/'.format( next_dir_number )
neural_net_dir_number_name = neural_net_directory_name + '{0:04d}/'.format( next_dir_number )
output_dir_number_name = output_directory_name + '{0:04d}/'.format( next_dir_number )

### Create directories.
if not os.path.isdir( data_dir_number_name ):
    os.makedirs( data_dir_number_name )

if not os.path.isdir( neural_net_dir_number_name ):
    os.makedirs( neural_net_dir_number_name )

if not os.path.isdir( output_dir_number_name ):
    os.makedirs(output_dir_number_name)



class layer_class:
    def __init__( self,
                  expression = 0,
                  layer_type = 0,
                  output_dimensionality = 0,
                  kernel_x_ratio = 0,
                  kernel_y_ratio = 0,
                  stride_x_ratio = 0,
                  stride_y_ratio = 0,
                  act = 0,
                  use_bias = 0,
                  bias_init = 0,
                  bias_reg = 0,
                  bias_reg_l1l2_type = 0,
                  act_reg = 0,
                  act_reg_l1l2_type = 0,
                  kernel_init = 0,
                  kernel_reg = 0,
                  kernel_reg_l1l2_type = 0,
                  dropout_rate = 0,
                  pool_x_ratio = 0,
                  pool_y_ratio = 0,
                  padding = 0 ):

        self.expression = expression
        self.layer_type = layer_type
        self.output_dimensionality = output_dimensionality
        self.kernel_x_ratio = kernel_x_ratio
        self.kernel_y_ratio = kernel_y_ratio
        self.stride_x_ratio = stride_x_ratio
        self.stride_y_ratio = stride_y_ratio
        self.act = act
        self.use_bias = use_bias
        self.bias_init = bias_init
        self.bias_reg = bias_reg
        self.bias_reg_l1l2_type = bias_reg_l1l2_type
        self.act_reg = act_reg
        self.act_reg_l1l2_type = act_reg_l1l2_type
        self.kernel_init = kernel_init
        self.kernel_reg = kernel_reg
        self.kernel_reg_l1l2_type = kernel_reg_l1l2_type
        self.dropout_rate = dropout_rate
        self.pool_x_ratio = pool_x_ratio
        self.pool_y_ratio = pool_y_ratio
        self.padding = padding

    def set_attribute( self, attribute, value ):
        setattr( self, attribute, value )

    def get_attribute( self, attribute ):
        return getattr( self, attribute )

    @property
    def get_attributes( self ):
        return self.__dict__


### Get the fitness of an individual.
def evaluate( individual, g, original_x_dimension, original_y_dimension ):
    ID = individual.ID

    ### Convert genotype to phenotype.
    phenotype_list = []
    num_layers = len( individual )
    #zero_layers = 0                      ### Keep track of how many empty layers ('0' layers) there are in the network.
    first_expressed_layer_added = False

    previous_x_dimension = original_x_dimension
    previous_y_dimension = original_y_dimension

    for n, layer in enumerate( individual ):
        ### If layer.expression == 1, this layer is an expressed chromosome.
        if layer.expression == 1:
            phenotype = get_phenotype( layer, first_expressed_layer_added, previous_x_dimension, previous_y_dimension )
            first_expressed_layer_added = True

            if layer.layer_type in layers_with_kernel or layer.layer_type in layers_with_pooling:
                previous_x_dimension, previous_y_dimension = compute_new_dimensions( layer.kernel_x_ratio, layer.kernel_y_ratio, layer.stride_x_ratio, layer.stride_y_ratio, previous_x_dimension, previous_y_dimension )
            

        ### If layer.expression == 0, this layer is an empty layer.
        elif layer.expression == 0:
            #zero_layers += 1
            phenotype = '##### ----- EMPTY LAYER ----- ##### ... ' + get_phenotype( layer, first_expressed_layer_added, previous_x_dimension, previous_y_dimension )

        ''' 
        Leaving out expression = 2 and 3 for now ... Only have a layer expressed or not expressed.
        ### If layer.expression == 2, this layer is a dropout layer.
        elif layer.expression == 2:
            phenotype = 'model.add( Dropout({}) )'.format( layer.dropout_rate )

        ### If layer.expression == 3, this layer is a flatten layer.
        elif layer.expression == 3:
            phenotype = 'model.add( Flatten() )'
        '''

        phenotype_list.append( phenotype )        ### List of layers in text format.

    ### Check to make sure there are non-empty layers.
    #if zero_layers != num_layers:
    ### Create directories to store the neural networks and the output from the genetic algorithm.
    neural_net_directory_name = 'neural_network_files/{0}/{0}{1:02d}/{0}{1:02d}{2:02d}/{3:04d}/'.format( year, month, day, next_dir_number )
    output_directory_name = 'output_files/{0}/{0}{1:02d}/{0}{1:02d}{2:02d}/{3:04d}/'.format( year, month, day, next_dir_number )

    neural_net_generation_dir_name = neural_net_directory_name + 'generation_{0:05d}/'.format( g )
    output_generation_dir_name = output_directory_name + 'generation_{0:05d}/'.format( g )

    ### Create directories if they do not exist.
    if not os.path.isdir( neural_net_generation_dir_name ):
        try:
            os.makedirs( neural_net_generation_dir_name )
        except:
            pass

    if not os.path.isdir( output_generation_dir_name ):
        try:
            os.makedirs( output_generation_dir_name )
        except:
            pass

    ### Specific name for neural network file and output file.
    run_file_name = neural_net_generation_dir_name + '{0}{1:02d}{2:02d}_individual_{3:04d}_neural_net.py'.format( year, month, day, ID )
    output_file_name = output_generation_dir_name + '{0}{1:02d}{2:02d}_individual_{3:04d}_output.txt'.format( year, month, day, ID )

    ### Open neural network file and write in some lines.
    with open( run_file_name, 'w' ) as run_file:
        run_file.write( 'batch_size = ' + str(batch_size) + '\n')
        run_file.write( 'number_of_classes = ' + str(number_of_classes) + '\n')
        run_file.write( 'epochs = ' + str(epochs) + '\n')
        ### Write top wrapper to file.
        for line in top_lines:
            run_file.write( line.split('\n')[0] + '\n')

        ### Write phenotype to file.
        for phenotype in phenotype_list:
            #print('phenotype: ', phenotype)
            run_file.write( phenotype + '\n' )

        ### Write bottom wrapper to file.
        for line in bot_lines:
            run_file.write( line.split('\n')[0] + '\n' )
    
    ### Save time at which the job was started.
    #start = time.time()

    ### Start the job.
    proc = subprocess.Popen( ['python3.6', run_file_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE )

    ### Save time at which the job ended.
    #end = time.time()

    ### Compute the runtime of the job.
    #duration = end-start

    ### Compute the duration fitness based on how long it took to run the job.
    #inverse_duration = 1./duration

    ### Capture the output of the job.
    out = proc.communicate()[0].decode( 'utf-8' )
    #print('ID: {}, out: {}'.format(ID, out) )
    #print(out.upper().split()[-2], out.upper().split()[-1])
    #print(out.upper().split()[-4], out.upper().split()[-3])
    try:
        ### Get the accuracy.
        accuracy = float( out.upper().split()[-1] )
        ### Compute the inverse loss.
        inverse_loss = 1./float( out.upper().split()[-3] )
    except:
        accuracy = 0
        inverse_loss = 100

    #inverse_mem = 1./float(out.upper().split()[-3])
    #inverse_cpu = 1./float(out.upper().split()[-1])

    ### Collect the fitness values.
    #fitness = ( accuracy, inverse_loss, inverse_duration, inverse_mem, inverse_cpu )
    fitness = [ accuracy, inverse_loss ]

    ### Return fitness of 0 if the neural network file did not complete a run for whatever reason.
    #else:
    #   fitness = [ 0, 0 ]

    ### Return the fitness value for the individual..
    return fitness

### Define how individuals mutate.
def mutation( individual, x_dimension_length, y_dimension_length ):
    ### Start a list to hold the mutated layers.
    mutated_individual = []

    ### Iterate over the layers of the individual.
    for c, layer in enumerate( individual ):
        ### Mutate chromosome (layer) expression. The possible values are 0 (not expressed), 1 (expressed), 2 (dropout layer), 3 (flatten).

        if c != 0:
            r = np.random.uniform( 0, 1 )
            if r <= layer_expression_rate:
                layer.expression = 1
            else:
                layer.expression = 0

        else:
            layer.expression = 1

        ''' 
        r = np.random.uniform( 0, 1 )
        if r <= mutation_probability:
            ### If the layer is the first layer, we must have it expressed as an actual layer (instead of empty, flatten, or dropout).
            if c != 0:
                #layer.expression = np.random.randint( 4 )
                layer.expression = np.random.randint( 2 )
            else:
                layer.expression = 1
        '''
        ### Mutate layer type. The value is in the range 0:5 (inclusive).
        r = np.random.uniform( 0, 1 )
        if r <= mutation_probability:
            layer.layer_type = np.random.randint( 6 )

        ### Mutate the output dimensionality.
        r = np.random.uniform( 0, 1 )
        if r <= mutation_probability:
            layer.output_dimensionality = int( np.random.normal( layer.output_dimensionality, 1 ) )

            ### Turn the output dimension to 2 if it is less than 2. This could be changed later.
            ### CHANGE THIS
            if layer.output_dimensionality < 2:
                layer.output_dimensionality = 2

        ### // Out-dated: Mutate kernel x length.
        ### Mutate the kernel x length ratio.
        r = np.random.uniform( 0, 1 )
        if r <= mutation_probability:
            #kernel_x_length = layer.kernel_x
            #ratio = float(kernel_x_length) / x_dimension_length
            #ratio = layer.kernel_x_ratio
            new_ratio = np.random.normal( layer.kernel_x_ratio, .1 )
            ### Make the ratios non-negative. Make the ratios equal to 1 if they are greater than 1.
            if ratio < 0:
                ratio = 0
            elif ratio > 1:
                ratio = 1
            #new_kernel_x_length = int( new_ratio * x_dimension_length )

            #layer.kernel_x = new_kernel_x_length
            layer.kernel_x_ratio = ratio

        ### // Out-dated: Mutate kernel y length.
        ### Mutate the kernel y length ratio.
        r = np.random.uniform( 0, 1 )
        if r <= mutation_probability:
            #kernel_y_length = layer.kernel_y
            #ratio = float(kernel_y_length) / y_dimension_length
            #ratio = layer.kernel_y_ratio
            new_ratio = np.random.normal( layer.kernel_y_ratio, .1 )
            ### Make the ratios non-negative. Make the ratios equal to 1 if they are greater than 1.
            if new_ratio < 0:
                new_ratio = 0
            elif new_ratio > 1:
                new_ratio = 1
            #new_kernel_y_length = int( new_ratio * y_dimension_length )

            #layer.kernel_y = new_kernel_y_length
            layer.kernel_y_ratio = ratio

        ### // Out-dated: Mutate stride length.
        ### Mutate the stride length in the x dimension.
        ###     When mutating the stride length, the ratio is taken between the dimension
        ###     length and the stride length. The ratio is mutated and the new ratio is
        ###     used to compute the new strides length.

        ### Mutate the stride x ratio.
        r = np.random.uniform( 0, 1 )
        if r <= mutation_probability:
            #stride_x_length = layer.stride_x
            #ratio = float(stride_x_length) / x_dimension_length
            #ratio = layer.stride_x_ratio
            new_ratio = np.random.normal( layer.stride_x_ratio, .1 )
            if new_ratio > 1:
                new_ratio = 1
            elif new_ratio < 0:
                new_ratio = .1

            #new_stride_x_length = int( new_ratio * x_dimension_length )

            layer.stride_x_ratio = new_ratio


        ### // Out-dated: Mutate stride length.
        ### Mutate the stride length in the y dimension.
        ###     See the comment for stride_x_length.

        ### Mutate the stride y ratio.
        r = np.random.uniform( 0, 1 )
        if r <= mutation_probability:
            #stride_y_length = layer.stride_y
            #ratio = float(stride_y_length) / y_dimension_length
            #ratio = layer.stride_y_ratio
            new_ratio = np.random.normal( layer.stride_y_ratio, .1 )
            if new_ratio > 1:
                new_ratio = 1
            elif new_ratio < 0:
                new_ratio = .1

            #new_stride_y_length = int( new_ratio * y_dimension_length )

            layer.stride_y_ratio = new_ratio

        ### Mutate activation type.
        r = np.random.uniform( 0, 1 )
        if r <= mutation_probability:
            layer.act = np.random.randint( 0, 11 )

        ### Mutate use bias.
        r = np.random.uniform( 0, 1 )
        if r <= mutation_probability:
            layer.use_bias = np.random.randint( 0, 2 )

        ### Mutate bias initializer.
        r = np.random.uniform( 0, 1 )
        if r <= mutation_probability:
            layer.bias_init = np.random.randint( 0, 11 )

        ### Mutate bias regularizer.
        r = np.random.uniform( 0, 1 )
        if r <= mutation_probability:
            new_bias_reg = np.random.normal( layer.bias_reg, .1 )
            if new_bias_reg > 1:
                new_bias_reg = 1
            elif new_bias_reg < -1:
                new_bias_reg = -1
            layer.bias_reg = new_bias_reg

        ### Mutate bias l1l2 regularizer type.
        r = np.random.uniform( 0, 1 )
        if r <= mutation_probability:
            new_bias_reg_l1l2_type = np.random.randint( 0, 3 )
            layer.bias_reg_l1l2_type = new_bias_reg_l1l2_type

        ### Mutate activity regularizer.
        r = np.random.uniform( 0, 1 )
        if r <= mutation_probability:
            #layer[11] = np.random.normal( layer[11], .1 )
            new_act_reg = np.random.normal( layer.act_reg, .1 )
            if new_act_reg > 1:
                new_act_reg = 1
            elif new_act_reg < -1:
                new_act_reg = -1
            layer.act_reg = new_act_reg

        ### Mutate activity l1l2 regularizer type.
        r = np.random.uniform( 0, 1 )
        if r <= mutation_probability:
            new_act_reg_l1l2_type = np.random.randint( 0, 3 )
            layer.act_reg_l1l2_type = new_act_reg_l1l2_type

        ### Mutate kernel initializer.
        r = np.random.uniform( 0, 1 )
        if r <= mutation_probability:
            layer.kernel_init = np.random.randint( 0, 11 )

        ### Mutate kernel regularizer.
        r = np.random.uniform( 0, 1 )
        if r <= mutation_probability:
            new_kernel_reg = np.random.normal( layer.kernel_reg, .1 )
            if new_kernel_reg > 1:
                new_kernel_reg = 1
            elif new_kernel_reg < -1:
                new_kernel_reg = -1
            layer.kernel_reg = new_kernel_reg

        ### Mutate kernel l1l2 regularizer type.
        r = np.random.uniform( 0, 1 )
        if r <= mutation_probability:
            new_kernel_reg_l1l2_type = np.random.randint( 0, 3 )
            layer.kernel_reg_l1l2_type = new_kernel_reg_l1l2_type

        ### Mutate dropout rate.
        r = np.random.uniform( 0, 1 )
        if r <= mutation_probability:
            new_dropout_rate = np.random.normal( layer.dropout_rate, .1 )
            if new_dropout_rate > 1:
                new_dropout_rate = 1
            elif new_dropout_rate < 0:
                new_dropout_rate = 0
            layer.dropout_rate = new_dropout_rate

        ### // Out-dated: Mutate pool length in the x dimension.
        ### Mutate pool x ratio.
        r = np.random.uniform( 0, 1 )
        if r <= mutation_probability:
            #layer[14] = np.random.randint( 1, 11 )
            #layer.pool_size_x = np.random.randint( 1, 11 )
            #ratio = layer.pool_x_ratio
            new_ratio = np.random.normal( layer.pool_x_ratio, .1 )
            if new_ratio > 1:
                new_ratio = 1
            elif new_ratio < 0:
                new_ratio = .1

            #new_stride_y_length = int( new_ratio * y_dimension_length )

            layer.pool_x_ratio = new_ratio

        ### // Out-dated: Mutate pool length in the y dimension.
        ### Mutate pool y ratio.
        r = np.random.uniform( 0, 1 )
        if r <= mutation_probability:
            #layer.pool_size_y = np.random.randint( 1, 11 )
            new_ratio = np.random.normal( layer.pool_y_ratio, .1 )
            if new_ratio > 1:
                new_ratio = 1
            elif new_ratio < 0:
                new_ratio = .1

            layer.pool_y_ratio = new_ratio

        ### Mutate padding type.
        r = np.random.uniform( 0, 1 )
        if r <= mutation_probability:
            layer.padding = np.random.randint( 0, 2 )

        ### Add the new mutated layer to the list representing the mutated individual.
        mutated_individual.append( layer )

    mutated_individual = creator.Individual( mutated_individual )

    return mutated_individual

''' 
################################ !!! OUT-DATED !!! ################################
### Check if the x and y kernals are valid (they should
###     be smaller than the corresponding dimension size.)
def check_kernel_validity( individual, original_x_dimension, original_y_dimension ):
    ### Set previous x and y dimensions equal to the original x and y dimensions.
    ###     (the x and y dimensions of the data set before the first layer.)
    previous_x_dimension = original_x_dimension
    previous_y_dimension = original_y_dimension

    ### Start a list for the modified individual. (The modified individual has its kernal sizes
    ###     modified if the kernal size is invalid. Saves the unmodified individual if the kernal
    ###     size is valid.)
    modified_individual = []

    ### Iterate over the layers.
    for c, layer in enumerate( individual ):
        padding_type = layer.padding
        expression = layer.expression
        print('layer num: {}, expression: {}, padding_type: {}'.format(c, expression, padding_type))

        #kernel_x_length = layer[3]
        #kernel_y_length = layer[4]
        #kernel_x_length = layer.kernel_x
        #kernel_y_length = layer.kernel_y

        kernel_x_length = layer.kernel_x_ratio * previous_x_dimension
        kernel_y_length = layer.kernel_y_ratio * previous_y_dimension

        stride_x_length = layer.stride_x
        stride_y_length = layer.stride_y

        if padding_type == 'valid':
            ########## UPDATE THIS TEXT ##########
            ### Check if the kernal size is greater than the dimension size. The kernel size
            ###     needs to be less than or equal to the dimension size minus 1 (for stride = 1).
            ###     output = input - (kernel - 1).
            ###     If the kernal size is too large, generate a random number between 0 and 1 and
            ###     take the floor of [that number times the dimension size]. This gives a kernal size
            ###     that is less than the dimension size.
            ######################################

            #print('layer: {}\nkernel_x_length: {}, previous_x_dimension - 1: {}\nkernel_y_length: {}, previous_y_dimension - 1: {}\n'.format( c, kernel_x_length, previous_x_dimension-1, kernel_y_length, previous_y_dimension-1 ))

            if kernel_x_length > previous_x_dimension:
                kernel_x_ratio = np.random.uniform( 0, 1 )
                kernel_x_length = int( math.floor( kernel_x_ratio * previous_x_dimension ) )

            if kernel_y_length > previous_y_dimension:
                kernel_y_ratio = np.random.uniform( 0, 1 )
                kernel_y_length = int( math.floor( kernel_y_ratio * previous_y_dimension ) )

            ### Check if the kernel length is less than 1. If so, change the size to be equal to 1.
            if kernel_x_length < 1:
                kernel_x_length = 1
            if kernel_y_length < 1:
                kernel_y_length = 1

            ### New dimension size is the old dimension size minus the quantity
            ###     kernel size minus 1.

            #previous_x_dimension -= (kernel_x_length - 1)
            #previous_y_dimension -= (kernel_y_length - 1)

            new_x_dimension = math.floor( ( previous_x_dimension - kernel_x_length + stride_x_length ) / stride_x_length )
            new_y_dimension = math.floor( ( previous_y_dimension - kernel_y_length + stride_y_length ) / stride_y_length )

            if new_x_dimension < 1:
                previous_x_dimension = 1
                #kernel_x_length = 1

            if new_y_dimension < 1:
                previous_y_dimension = 1
                #kernel_y_length = 1

            ### Save the new, valid kernel sizes to the chunk (layer).
            #layer[3] = int( math.floor( kernel_x_length ) )
            #layer[4] = int( math.floor( kernel_y_length ) )
            layer.kernel_x = int( math.floor( kernel_x_length ) )
            layer.kernel_y = int( math.floor( kernel_y_length ) )

        elif padding == 'same':
            px = ( math.ceil( previous_x_dimention / stride_x_length ) - 1 )*stride_x_length + kernel_x_length - previous_x_dimension
            new_x_dimension = math.floor( ( previous_x_dimension - kernel_x_length + px + stride_x_length ) / stride_x_length )

            py = ( math.ceil( previous_y_dimention / stride_y_length ) - 1 )*stride_y_length + kernel_y_length - previous_y_dimension
            new_y_dimension = math.floor( ( previous_y_dimension - kernel_y_length + py + stride_y_length ) / stride_y_length )

            previous_x_dimension = new_x_dimension
            previous_y_dimension = new_y_dimension

        print('(new) previous_x_dimension: {}, (new) previous_y_dimension: {}'.format(previous_x_dimension, previous_y_dimension))


        #print('layer: {}\nnew kernel_x_length: {}, previous_x_dimension - 1: {}\nnew kernel_y_length: {}, previous_y_dimension - 1: {}\n'.format( c, kernel_x_length, previous_x_dimension-1, kernel_y_length, previous_y_dimension-1 ))

        ### Save modified chunk to the new chromosome.
        modified_individual.append( layer )

    ### Create the modified individual using keras.creator.Individual.
    modified_individual = creator.Individual( modified_individual )

    return modified_individual
################################ !!! OUT-DATED !!! ################################
'''

def compute_new_dimensions( kernel_x_ratio, kernel_y_ratio, stride_x_ratio, stride_y_ratio, x_dimension, y_dimension ):
    kernel_x = kernel_x_ratio * x_dimension
    kernel_y = kernel_y_ratio * y_dimension

    stride_x = stride_x_ratio * x_dimension
    stride_y = stride_y_ratio * y_dimension

    output_x = math.floor( ( x_dimension - kernel_x + stride_x ) / stride_x )
    output_y = math.floor( ( y_dimension - kernel_y + stride_y ) / stride_y )

    return output_x, output_y


def compute_layer_dimensions( individual, original_x_dimension, original_y_dimension ):
    previous_x_dimension = original_x_dimension
    previous_y_dimension = original_y_dimension

    for c, layer in enumerate( individual ):
        layer_type = layer.layer_type

        if ( (layer_type in layers_with_kernel) or (layer_type in layers_with_pooling) ):
            kernel_x_ratio = layer.kernel_x_ratio
            kernel_y_ratio = layer.kernel_y_ratio
            stride_x_ratio = layer.stride_x_ratio
            stride_y_ratio = layer.stride_y_ratio
        
            previous_x_dimension, previous_y_dimension = compute_new_dimensions( kernel_x_ratio, kernel_y_ratio, stride_x_ratio, stride_y_ratio, previous_x_dimension, previous_y_dimension )

            #print('c:{}, previous_x_dimension: {}, previous_y_dimension: {}'.format(c, previous_x_dimension, previous_y_dimension))


def test_compute_layer_dimensions():
    original_x_dimension = 32
    original_y_dimension = 32

    individual = [ layer_class() for c in range(5) ]

    individual[0].layer_type = 1
    individual[0].padding = 0
    individual[0].kernel_x_ratio = 2/32
    individual[0].kernel_y_ratio = 1/32
    individual[0].stride_x_ratio = 2/32
    individual[0].stride_y_ratio = 1/32
    
    individual[1].layer_type = 1
    individual[1].padding = 0
    individual[1].kernel_x_ratio = 1/16
    individual[1].kernel_y_ratio = 2/32
    individual[1].stride_x_ratio = 1/16
    individual[1].stride_y_ratio = 2/32

    individual[2].layer_type = 1
    individual[2].padding = 0
    individual[2].kernel_x_ratio = 3/16
    individual[2].kernel_y_ratio = 4/16
    individual[2].stride_x_ratio = 4/16
    individual[2].stride_y_ratio = 3/16

    individual[3].layer_type = 1
    individual[3].padding = 0
    individual[3].kernel_x_ratio = 4/4
    individual[3].kernel_y_ratio = 3/5
    individual[3].stride_x_ratio = 3/4
    individual[3].stride_y_ratio = 4/5

    compute_layer_dimensions( individual, original_x_dimension, original_y_dimension )



def expression():                        ### Return 0 (skip layer), 1 (use layer), 2 (dropout layer), 3 (flatten layer).
    #return np.random.randint(0, 4)
    return np.random.randint( 0, 2 )

def layer_type():                       ### Return random integer between 0 and 5 for layer type.
    return np.random.randint( 6 )

def output_dimensionality():            ### Return random integer between 2 and 100 for number of output_dimensionality for layer.
    return np.random.randint( 2, 101 )

################ OUT-DATED ################
#def kernel_x( x_dimension_length ):
#def kernel_x():                    ### Return kernel size for x dimension (THIS IS NOT RIGHT --> as a fraction of the dimension length).
                                        ###    The kernel length is an integer in the actual chromosome. When doing mutations, the ratio is
                                        ###    computed using the kernel length and the dimension length of the current layer.
    #return np.random.randint( 1, x_dimension_length + 1 )
#def kernel_y( y_dimension_length ):
#def kernel_y():                    ### Return kernel size for y dimension (SEE ABOVE --> as a fraction of the dimension length).
    #return np.random.randint( 1, y_dimension_length + 1 )
###########################################

def kernel_x_ratio():
    return np.random.uniform()

def kernel_y_ratio():
    return np.random.uniform()

def stride_x_ratio():
    return np.random.randint( 1, 11 )

def stride_y_ratio():
    return np.random.randint( 1, 11 )

def act():                              ### Return random integer between 0 and 10 for layer activation type.
    return np.random.randint( 11 )

def use_bias():                         ### Return random integer between 0 and 1 for use_bias = True or False.
    return np.random.randint( 2 )

def bias_init():                        ### Return random integer between 0 and 10 for bias initializer for layer.
    return np.random.randint( 11 )

def bias_reg():                         ### Return random float between 0 and 1 for bias regularizer for layer.
    return np.random.uniform()

def bias_reg_l1l2_type():
    return np.random.randint( 0, 3 )

def act_reg():                          ### Return random float between 0 and 1 for activation regularizer for layer.
    return np.random.uniform()

def act_reg_l1l2_type():
    return np.random.randint( 0, 3 )

def kernel_init():                      ### Return random integer between 0 and 10 for the type of kernel initializer.
    return np.random.randint( 11 )

def kernel_reg():
    return np.random.uniform()

def kernel_reg_l1l2_type():
    return np.random.randint( 0, 3 )

def dropout_rate():                     ### Return random float between 0 and 1 for the dropout rate.
    return np.random.uniform()

def pool_x_ratio():
    return np.random.randint( 1, 11 )

def pool_y_ratio():
    return np.random.randint( 1, 11 )

def padding():
    return np.random.randint( 0, 2 )

def layervalue():
    return np.random.randint( 10, 40 )


def generate_individual( num_layers, x_dimension_length, y_dimension_length ):
    individual = []
    for n in range(num_layers):
        layer = layer_class( expression(),
                             layer_type(),
                             output_dimensionality(),
                             kernel_x_ratio(),
                             kernel_y_ratio(),
                             stride_x(),
                             stride_y(),
                             act(),
                             use_bias(),
                             bias_init(),
                             bias_reg(),
                             bias_reg_l1l2_type(),
                             act_reg(),
                             act_reg_l1l2_type(),
                             kernel_init(),
                             kernel_reg(),
                             kernel_reg_l1l2_type(),
                             dropout_rate(),
                             pool_x_ratio(),
                             pool_y_ratio(),
                             padding() )

        individual.append( layer )

    return individual


### Convert x to an int or a float.
def convert( x ):
    try:
        return int( x )
    except ValueError:
        return float( x )

### Extract the initial population
def seed_population( initial_population_directory ):
    init_population = []
    for fil in os.listdir(initial_population_directory):
        file_name = initial_population_directory + '/' + fil

        chromosome = []
        with open(file_name) as txt_file:
            lines = txt_file.readlines()

            for line in lines:
                converted_fields = []
                fields = line.split()

                for field in fields:
                    field = convert( field.strip(',') )
                    converted_fields.append( field )

                expression = converted_fields[0]
                layer_type = convereted_fields[1]
                output_dimensionality = convereted_fields[2]
                kernel_x_ratio = convereted_fields[3]
                kernel_y_ratio = convereted_fields[4]
                stride_x_ratio = convereted_fields[5]
                stride_y_ratio = convereted_fields[6]
                act = convereted_fields[7]
                use_bias = convereted_fields[8]
                bias_init = convereted_fields[9]
                bias_reg = convereted_fields[10]
                bias_reg_l1l2_type = convereted_fields[11]
                act_reg = convereted_fields[12]
                act_reg_l1l2_type = convereted_fields[13]
                kernel_init = convereted_fields[14]
                kernel_reg = convereted_fields[15]
                kernel_reg_l1l2_type = convereted_fields[16]
                dropout_rate = convereted_fields[17]
                pool_x_ratio = convereted_fields[18]
                pool_y_ratio = convereted_fields[19]
                padding = convereted_fields[20]

                layer = layer_class( expression,
                                     layer_type,
                                     output_dimensionality,
                                     kernel_x_ratio,
                                     kernel_y_ratio,
                                     stride_x_ratio,
                                     stride_y_ratio,
                                     act,
                                     use_bias,
                                     bias_init,
                                     bias_reg,
                                     bias_reg_l1l2_type,
                                     act_reg,
                                     act_reg_l1l2_type,
                                     kernel_init,
                                     kernel_reg,
                                     kernel_reg_l1l2_type,
                                     dropout_rate,
                                     pool_x_ratio,
                                     pool_y_ratio,
                                     padding )

                chromosome.append( layer )

        init_population.append( chromosome )

    return init_population


### Crossover between two parents. Creates two children.
def crossover( parent1, parent2, crossover_probability ):
    child1 = list( parent1 )
    child2 = list( parent2 )

    attributes = parent1[0].get_attributes

    for m, layer in enumerate( parent1 ):
        for attribute in attributes:
            r = np.random.uniform( 0, 1 )
            if r <= crossover_probability:
                attribute1 = parent1[m].get_attribute( attribute )
                attribute2 = parent2[m].get_attribute( attribute )

                child1[m].set_attribute( attribute, attribute2 )
                child2[m].set_attribute( attribute, attribute1 )

                #child1[m].type = 'CHILD'
                #child2[m].type = 'CHILD'

    return child1, child2


### Extract training data.
with open('../x_train.pkl', 'rb') as pkl_file:
    x_train = pickle.load( pkl_file, encoding = 'latin1' )

### Get dimension of data (size of x and y dimensions).
original_x_dimension = x_train.shape[1]
original_y_dimension = x_train.shape[2]
previous_x_dimension = original_x_dimension
previous_y_dimension = original_y_dimension


creator.create( 'FitnessMax', base.Fitness, weights = (1., 1.) )
creator.create( 'Individual', list, fitness = creator.FitnessMax, ID = 0, type = 'NA' )


toolbox = base.Toolbox()
### Create a string of a command to be executed to register a 'type' of individual.
toolbox_ind_str = "toolbox.register('individual', tools.initCycle, creator.Individual, ("

''' 
# This is not needed and is out-of-date.
### Iterate over the number of layers and append to the string to be executed.
for n in range( max_num_layers ):
    expression_str = 'expression_' + str(n)
    layer_str = 'layer_type_' + str(n)
    output_dimensionality_str = 'output_dimensionality_' + str(n)
    kernel_x_str = 'kernel_x_' + str(n)
    kernel_y_str = 'kerner_y_' + str(n)
    stride_x_str = 'stride_x_' + str(n)
    stride_y_str = 'stride_y_' + str(n)
    act_str = 'act_' + str(n)
    use_bias_str = 'use_bias_' + str(n)
    bias_init_str = 'bias_init_' + str(n)
    bias_reg_str = 'bias_reg_' + str(n)
    act_reg_str = 'act_reg_' + str(n)
    kernel_init_str = 'kernel_init_' + str(n)
    dropout_rate_str = 'dropout_rate_' + str(n)
    pool_size_x_str = 'pool_size_x_' + str(n)
    pool_size_y_str = 'pool_size_y_' + str(n)
    padding_str = 'padding_' + str(n)

    toolbox.register(expression_str, expression)
    toolbox.register(layer_str, layer_type)
    toolbox.register(output_dimensionality_str, output_dimensionality)

    toolbox.register( kernel_x_str, kernel_x )
    toolbox.register( kernel_y_str, kernel_y )
    toolbox.register( stride_x_str, stride_x )
    toolbox.register( stride_y_str, stride_y )
    toolbox.register( act_str, act )
    toolbox.register( use_bias_str, use_bias )
    toolbox.register( bias_init_str, bias_init )
    toolbox.register( bias_reg_str, bias_reg )
    toolbox.register( act_reg_str, act_reg )
    toolbox.register( kernel_init_str, kernel_init )
    toolbox.register( dropout_rate_str, dropout_rate )
    toolbox.register( pool_size_x_str, pool_size_x )
    toolbox.register( pool_size_y_str, pool_size_y )
    toolbox.register( padding_str, padding )

    toolbox_ind_str += 'toolbox.' + expression_str + ', toolbox.' + layer_str
    toolbox_ind_str += ', toolbox.' + output_dimensionality_str + ', toolbox.' + kernel_x_str
    toolbox_ind_str += ', toolbox.' + kernel_y_str + ', toolbox.' + stride_x_str
    toolbox_ind_str += ', toolbox.' + stride_y_str + ', toolbox.' + act_str
    toolbox_ind_str += ', toolbox.' + use_bias_str + ', toolbox.' + bias_init_str
    toolbox_ind_str += ', toolbox.' + bias_reg_str + ', toolbox.' + act_reg_str
    toolbox_ind_str += ', toolbox.' + kernel_init_str + ', toolbox.' + dropout_rate_str
    toolbox_ind_str += ', toolbox.' + pool_size_x_str + ', toolbox.' + pool_size_y_str
    toolbox_ind_str += ', toolbox.' + padding_str

    if n != max_num_layers-1:
        toolbox_ind_str += ", "
'''
toolbox_ind_str += "), n=1)"

### Execute string to register individual type.
exec( toolbox_ind_str )

### Register population, mutate, and select functions.
toolbox.register( 'population', tools.initRepeat, list, toolbox.individual )
# THIS ISN'T WORKING --> toolbox.register('mate', tools.cxUniform, crossover_probability)
toolbox.register( 'mutate', mutation )
toolbox.register( 'select', tools.selNSGA2 )

def main():

    #compute_layer_dimensions()

    ''' 
    print('\n... Running genetic algorithm on neural networks ...\n')

    print('max_number_of_layers: {}'.format( max_num_layers ) )

    ### Population size. Specified in inFile.txt.
    print('population_size: {}'.format( population_size ) )

    ### Number of individuals (parents) to clone for the next generation. Specified in inFile.txt.
    print('selection_size (number of parents): {}'.format( selection_size ) )

    ### Number of individuals made through crossing of selected parents. Same as the number of parents.
    crossover_size = selection_size
    print('crossover_size (number of children generated, same as the number of parents): {}'.format( crossover_size ) )

    ### Number of immigrants. Computed from the population size minus the amount of parents plus the amount of children.
    #migration_size = population_size - 2 * selection_size
    print('migration_size: {}'.format( migration_size ) )

    ### mutation probability. Specified in inFile.txt.
    print('mutation_probability (for each gene): {}'.format( mutation_probability ) )

    ### Crossover probability (for uniform crossover). Specified in inFile.txt.
    print('crossover_probability (for each gene, for uniform crossover): {}'.format( crossover_probability ) )

    ### Number of generations.Specified in inFile.txt.
    print('number_of_generations: {}'.format( number_of_generations ) )

    ### Set up the initial population.
    ###     Write 'FALSE' in inFile.txt for initialize with a random population.
    ###     Give a directory in inFile.txt from which to get the initial population.
    false_list = ['FALSE', 'false', 'False']

    if initial_population_directory not in false_list:
        ### Generate initial set of individuals from the directory containing the zero-th generation of individuals.
        temp_pop = seed_population( initial_population_directory )

        ### Fill the remaining spots available for more individuals.
        remaining_pop_to_initialize = population_size - len( temp_pop )
        remaining_pop = [ generate_individual( max_num_layers, original_x_dimension, original_y_dimension ) for i in range( remaining_pop_to_initialize ) ]

        ### Merge these two populations together.
        temp_pop = temp_pop + remaining_pop
    else:
        ### If an initial population directory is not used, randomly generate all individuals.
        temp_pop = [ generate_individual( max_num_layers, original_x_dimension, original_y_dimension ) for i in range( population_size ) ]

    ### ID numbers are used to keep track of individuals.
    ID = 0

    pop = []
    for ind in temp_pop:
        x = creator.Individual( check_kernel_validity( ind, original_x_dimension, original_y_dimension ) )
        
        #print('x: {}'.format(x))
        x.ID = ID                         ### Assign ID number to individual.
        ID += 1                           ### Increment the ID number.
        #print('x.ID: {}'.format(x.ID))
        pop.append(x)

    #print('\n\nInitial population ...')
    #for c, i in enumerate(pop):
        #print('Individual {}: '.format(c) )
        #print(i)

    print('Computing fitnesses for generation 0 ...\n')

    index = np.arange( population_size )
    generation = [ 0 for x in range( population_size ) ]

    pool = ThreadPool( population_size )

    fitnesses = pool.starmap( evaluate, zip( pop, generation ) )

    pool.close()
    pool.join()

    print('Generation 0 fitnesses ...\n')
    #for fitness in fitnesses:
    #    print(fitness)

    ### Save fitness values to individuals.
    for ind, fitness in zip( pop, fitnesses ):
        ind.fitness.values = fitness

    for ind in pop:
        print( 'ID: {}, fitness: {}'.format( ind.ID, ind.fitness.values ) )

    #for i, ind in enumerate( pop ):
    #    ind.fitness.values = [i, 0]

    ### Create directory to save the data for the 0th generation.
    generation_dir_name = data_directory_name + '{0:04d}/generation_00000/'.format(next_dir_number)
    if not os.path.isdir(generation_dir_name):
        os.makedirs(generation_dir_name)
    '''
    ''' 
    ### Save the population of the 0th generation.
    generation_population_file_name = generation_dir_name + '{0}{1:02d}{2:02d}_{3:04d}_generation_00000_population.pkl'.format(year, month, day, next_dir_number)
    with open(generation_population_file_name, 'wb') as fil:
        pickle.dump(pop, fil)
    ### Save the fitnesses of the 0th generation.
    generation_fitness_file_name = generation_dir_name + '{0}{1:02d}{2:02d}_{3:04d}_generation_00000_fitness.pkl'.format(year, month, day, next_dir_number)
    with open(generation_fitness_file_name, 'wb') as fil:
        pickle.dump(fitnesses, fil)
    '''
    ''' 
    generation_population_file_name = generation_dir_name + '{0}{1:02d}{2:02d}_{3:04d}_generation_00000_population.txt'.format(year, month, day, next_dir_number)
    with open( generation_population_file_name, 'w' ) as fil:
        for ind in pop:
            fil.write( 'ID: {}, fitness: {}\n'.format( ind.ID, ind.fitness ) )
            fil.write( '{}\n\n'.format( ind[0].get_attributes ) )

    ### Iterate over the generations.
    for g in range( 1, number_of_generations ):
        ### Select the parents.
        selected_parents = toolbox.select( pop, selection_size )

        for parent in selected_parents:
            #print(parent.fitness.values)
            parent.type = 'PARENT'

        print('\nGeneration {} ... \n'.format(g))

        new_children = []

        ### Mate the parents to form new individuals (children).
        for i in range( 0, selection_size - migration_size, 2 ):
            #print('parent1: {}\nparent2: {}\n'.format( selected_parents[i], selected_parents[i+1] ) )

            child1, child2 = crossover( selected_parents[i], selected_parents[i+1], crossover_probability )

            #print('child1: {}\nchild2: {}\n'.format(child1, child2 ) )

            new_children.append(child1)
            new_children.append(child2)

        ### Mutate the newly generated children.
        print('\nMutating the new population (selected parents and their children) ...')
        for m, individual in enumerate( new_children ):
            mutated_ind = toolbox.mutate( individual, original_x_dimension, original_y_dimension )
            new_children[m] = mutated_ind

        ### Check the kernel validity of the individuals in the new (mutated) children.
        new_children = [ creator.Individual( check_kernel_validity(ind, original_x_dimension, original_y_dimension) ) for ind in new_children ]

        for child in new_children:
            child.type = 'CHILD'

        ### Add migrants to the new population.
        print('\nAdding randomly generated migrants to the new population ...')
        migrants = [ generate_individual( max_num_layers, original_x_dimension, original_y_dimension ) for m in range( migration_size ) ]

        mated_migrants = []

        #print('These parents suck and will be purged from the population after mating with immigrants.')
        #print('These parents will be used to mate with migrants and then be removed from the population.')
        for i in range( 0, migration_size, 2 ):
            #print('parent 1: {}\nparent 2: {}\n'.format( selected_parents[-i], selected_parents[-(i+1)] ))

            mated_migrant1, mated_migrant2 = crossover( selected_parents[-i], selected_parents[-(i+1)], crossover_probability )

            mated_migrants.append(mated_migrant1)
            mated_migrants.append(mated_migrant2)

        mated_migrants = [ creator.Individual( check_kernel_validity(ind, original_x_dimension, original_y_dimension) ) for ind in mated_migrants ]

        for migrant in mated_migrants:
            migrant.type = 'MATED MIGRANT'

        migrants = [ creator.Individual( check_kernel_validity(ind, original_x_dimension, original_y_dimension) ) for ind in migrants ]

        for migrant in migrants:
            migrant.type = 'MIGRANT'

        ### Check the kernel validity of the individuals in the new migrants.
        #migrants = [ creator.Individual( check_kernel_validity(ind, original_x_dimension, original_y_dimension) ) for ind in migrants ]

        #for migrant in migrants:
        #    migrant.type = 'MIGRANT'

        #new_population = new_children + migrants
        new_population = new_children + mated_migrants + migrants

        ### Check the kernel validity of the individuals in the new population.

        for new_ind in new_population:
            new_ind.ID = ID
            ID += 1

        generation = [ g for x in range( len( new_population ) ) ]

        ### Run the neural networks of the new individuals to compute their fitness.
        pool = ThreadPool( len(new_population) )

        new_fitnesses = pool.starmap( evaluate, zip(new_population, generation) )

        pool.close()
        pool.join()
 
        ### Assign the fitnesses to the new individuals.
        for ind, fitness in zip( new_population, new_fitnesses ):
            ind.fitness.values = fitness

        ### Create directory to save the data for the g-th generation.
        generation_dir_name = data_directory_name + '{0:04d}/generation_{1:05d}/'.format(next_dir_number, g)
        if not os.path.isdir(generation_dir_name):
            os.makedirs(generation_dir_name)

        generation_population_file_name = generation_dir_name + '{0}{1:02d}{2:02d}_{3:04d}_generation_{4:05d}_population.txt'.format(year, month, day, next_dir_number, g)

        ### Set pop to be the new population.
        pop = selected_parents[:-migration_size] + new_population

        with open(generation_population_file_name, 'w') as fil:
            for ind in pop:
                if ind.type == 'PARENT':
                    fil.write( '_PARENT_ ID: {}, fitness: {}\n'.format( ind.ID, ind.fitness ) )

                elif ind.type == 'CHILD':
                    fil.write( '_CHILD_ ID: {}, fitness: {}\n'.format( ind.ID, ind.fitness ) )

                elif ind.type == 'MIGRANT':
                    fil.write( '_MIGRANT_ ID: {}, fitness: {}\n'.format( ind.ID, ind.fitness ) )

                fil.write( '{}\n\n'.format( ind[0].get_attributes ) )

        #print('\nFinal new population in Generation {}...'.format(g) )
        #for c, ind in enumerate(pop):
        #    print('Individual {} _{}_:'.format(ind.ID, ind.type) )
        #    print(ind)


        print('Generation {} new fitnesses:\n'.format(g))
        #for fitness in fitnesses:
        #    print(fitness)

        for ind in pop:
            print('ID: {}, fitness: {}'.format(ind.ID, ind.fitness.values))

    '''
    pop, fitnesses = 0, 0
    ### Return the final population and final fitnesses.
    return pop, fitnesses


if __name__ == '__main__':
    pop, fitnesses = main()
