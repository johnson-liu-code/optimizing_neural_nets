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

from deap import base
from deap import creator
from deap import tools

from genotype_to_phenotype import get_phenotype
from functions.divide_chunks import divide_chunks

with open('inFile.txt', 'r') as fil:
#with open('test_inFile.txt', 'r') as fil:
    lines = fil.readlines()

max_num_layers = int(lines[0].split()[2])
population_size = int(lines[1].split()[2])
selection_size = int(lines[2].split()[2])
MUTPB = float(lines[3].split()[2])
NGEN = int(lines[4].split()[2])
batch_size = int(lines[5].split()[2])
num_classes = int(lines[6].split()[2])
epochs = int(lines[7].split()[2])
init_dir = lines[8].split()[2]

#print(max_num_layers, population_size, selection_size, MUTPB, NGEN)


### Set number of layers.
#max_num_layers = 20
#max_num_layers = 1
print('max_num_layers: ', max_num_layers)

### Top and bottom wrapper text to be used in making the neural network.
top_file = 'wrapper_text/top_text.txt'
bot_file = 'wrapper_text/bot_text.txt'

with open(top_file, 'r') as top:
    top_lines = top.readlines()
with open(bot_file, 'r') as bot:
    bot_lines = bot.readlines()

today = datetime.datetime.now()
year = today.year
month = today.month
day = today.day

data_directory_name = 'data/{0}/{0}{1:02d}/{0}{1:02d}{2:02d}/'.format(year, month, day)
neural_net_directory_name = 'neural_network_files/{0}/{0}{1:02d}/{0}{1:02d}{2:02d}/'.format(year, month, day)

if not os.path.isdir(data_directory_name):
    os.makedirs(data_directory_name)
if not os.path.isdir(neural_net_directory_name):
    os.makedirs(neural_net_directory_name)

if not os.listdir(data_directory_name):
    next_dir_number = 1

else:
    last_dir_name = sorted(os.listdir(data_directory_name))
    last_dir_number = int(last_dir_name[-1])
    next_dir_number = last_dir_number + 1

data_dir_number_name = data_directory_name + '{0:04d}/'.format(next_dir_number)
neural_net_dir_number_name = neural_net_directory_name + '{0:04d}/'.format(next_dir_number)

if not os.path.isdir(data_dir_number_name):
    os.makedirs(data_dir_number_name)
if not os.path.isdir(neural_net_dir_number_name):
    os.makedirs(neural_net_dir_number_name)

### Get the fitness of an individual.
#def evaluate(individual):
def evaluate(individual, i, g):
    print('individual:', individual)
    x_chunks = divide_chunks(individual, 13)    # There are 13 genes within each chromosome (layer).
    print('x_chunks: ', x_chunks)

    ### Convert genotype to phenotype.
    phenotype_list = []
    #for chunk in x_chunks:
    num_layers = len(x_chunks)
    zero_layers = 0
    for n in range(num_layers):
        chunk = x_chunks[n]
        if chunk[0] == 1:
            phenotype = get_phenotype(chunk)
        elif chunk[0] == 2:
            phenotype = 'model.add(Dropout({}))'.format(chunk[11])
        elif chunk[0] == 3:
            phenotype = 'mode.add(Flatten())'
        else:
            zero_layers += 1
            phenotype = ''
        phenotype_list.append(phenotype)        # List of chromosomes in text format.

    if zero_layers != num_layers:
        neural_net_directory_name = 'neural_network_files/{0}/{0}{1:02d}/{0}{1:02d}{2:02d}/{3:04d}/'.format(year, month, day, next_dir_number)

        neural_net_generation_dir_name = neural_net_directory_name + 'generation_{0:05d}/'.format(g)

        if not os.path.isdir(neural_net_generation_dir_name):
            os.makedirs(neural_net_generation_dir_name)

        #if not os.listdir(neural_net_directory_name):
        #    next_file_number = 1

        #else:
        #    last_file_name = sorted(os.listdir(neural_net_directory_name))
        #    last_file_number = int(last_file_name[-1].split('_')[1])
        #    next_file_number = last_file_number + 1

        temp_file_name = neural_net_generation_dir_name + '{0}{1:02d}{2:02d}_individual_{3:04d}_neural_net.py'.format(year, month, day, i)

        with open(temp_file_name, 'w') as tempfile:
            tempfile.write( 'batch_size = ' + str(batch_size) + '\n')
            tempfile.write( 'num_classes = ' + str(num_classes) + '\n')
            tempfile.write( 'epochs = ' + str(epochs) + '\n')
            ### Write top wrapper to file.
            for line in top_lines:
                tempfile.write( line.split('\n')[0] + '\n')

            ### Randomly choose the number of output_dimensionality in the first layer.
            #number_of_first_layer_output_dimensionality = np.random.randint(2, 100)
            #number_of_first_layer_output_dimensionality = 10

            ### Write first layer to file.
            #tempfile.write("model.add(Conv2D(" + str(number_of_first_layer_output_dimensionality) + ", (3, 3), padding='same', activation='relu', input_shape=x_train.shape[1:]))\n")

            ### Write phenotype to file.
            for phenotype in phenotype_list:
                #print('phenotype: ', phenotype)
                tempfile.write(phenotype + '\n')

            ### Write bottom wrapper to file.
            for line in bot_lines:
                tempfile.write( line.split('\n')[0] + '\n' )
    
        ### Save time at which job was started.
        #start = time.time()

        ### Start the job.
        #proc = subprocess.Popen(['srun', '--ntasks', '1', '--nodes', '1', 'python3.6', temp_file_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        proc = subprocess.Popen(['python3.6', temp_file_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        ### Save time at which job ended.
        #end = time.time()

        ### Compute the runtime of the job.
        #duration = end-start

        ### Compute the duration fitness based on how long it took to run the job.
        #inverse_duration = 1./duration

        ### Capture the output of the job.
        out = proc.communicate()[0].decode('utf-8')

        ### Compute inverse loss, inverse memory, and inverse cpu usage fitnesses.
        inverse_loss = 1./float(out.upper().split()[-7])
        #loss = float(out.upper().split()[-7])
        #inverse_mem = 1./float(out.upper().split()[-3])
        #inverse_cpu = 1./float(out.upper().split()[-1])

        ### Save the accuracy.
        accuracy = float(out.upper().split()[-5])

        ### Collect the fitness values.
        #fitness = ( accuracy, inverse_loss, inverse_duration, inverse_mem, inverse_cpu )
        fitness = ( accuracy, inverse_loss )

        #fitness = [1, 1, 1, 1, 1]
        #fitness = [1, 1]

        ### Return the fitness values.

    else:
       fitness = [0, 0]

    return fitness

### Define how individuals mutate.
def myMutation(individual):
    ### Divide individual's chromosome into chunks that represent each layer.
    #num_of_layers = individual[0]
    #individual_without_first_element = individual[1:] 
    chunks = divide_chunks(individual, 13)
    
    #mut_num_of_layers = tools.mutUniformInt([ num_of_layers ], 1, max_num_layers, MUTPB)

    ### Start a list for mutated individuals.
    mutated_individual = []
    #mutated_individual += [ mut_num_of_layers ]

    ### Iterate over individuals.
    for chunk in chunks:
        ### Mutate chromosome (layer) expression. The value is 0 (not expressed) or 1 (expressed).
        #chunk[0] = tools.mutUniformInt([ chunk[0] ], 0, 2, MUTPB)
        r = np.random.uniform(0, 1)
        if r <= MUTPB:        
            chunk[0] = np.random.randint(3)

        ### Mutate layer type.
        #chunk[1] = tools.mutUniformInt([ chunk[1] ], 0, 5, MUTPB)
        r = np.random.uniform(0, 1)
        if r <= MUTPB:
            chunk[1] = np.random.randint(6)

        ### Mutate number of output_dimensionality.
        #chunk[2] = tools.mutUniformInt([ chunk[2] ], 2, 4, MUTPB)
        #chunk[2] = tools.mutUniformInt([ chunk[2] ], 2, 10, MUTPB)
        r = np.random.uniform(0, 1)
        if r <= MUTPB:
            chunk[2] = np.random.randint(2, 11)

        ### Mutate kernel x number. (This is expressed as a fraction of the x dimension length.)
        #chunk[3] = tools.mutGaussian([ chunk[3] ], chunk[3], .1, MUTPB)
        r = np.random.uniform(0, 1)
        if r <= MUTPB:
            chunk[3] = np.random.normal(chunk[3], .1)

        ### Mutate kernel y number. (This is expressed as a fraction of the y dimension length.)
        #print('chunk4: ', chunk[4])
        #chunk[4] = tools.mutGaussian([ chunk[4] ], chunk[4], .1, MUTPB)
        r = np.random.uniform(0, 1)
        if r <= MUTPB:
            chunk[4] = np.random.normal(chunk[4], .1)

        #print('chunk4: ', chunk[4])

        ### Make the ratios positive.
        #if chunk[3][0][0] < 0:
        #    chunk[3][0][0] += 1
        #if chunk[4][0][0] < 0:
        #    chunk[4][0][0] += 1
        if chunk[3] < 0:
            chunk[3] += 1
        #print('chunk3: ', chunk[3], ' chunk4: ', chunk[4])
        if chunk[4] < 0:
            chunk[4] += 1

        ### Mutate strides.
        #chunk[5] = tools.mutUniformInt([ chunk[5] ], 1, 10, MUTPB)
        r = np.random.uniform(0, 1)
        if r <= MUTPB:
            chunk[5] = np.random.randint(1, 11)
        #print('\n strides: ' + str(chunk[5]) + '\n')

        ### Mutate activation type.
        #chunk[6] = tools.mutUniformInt([ chunk[6] ], 0, 10, MUTPB)
        r = np.random.uniform(0, 1)
        if r <= MUTPB:
            chunk[6] = np.random.randint(0, 11)

        ### Mutate use bias.
        #chunk[7] = tools.mutUniformInt([ chunk[7] ], 0, 1, MUTPB)
        r = np.random.uniform(0, 1)
        if r <= MUTPB:
            chunk[7] = np.random.randint(0, 2)

        ### Mutate bias initializer.
        #chunk[8] = tools.mutUniformInt([ chunk[8] ], 0, 10, MUTPB)
        r = np.random.uniform(0, 1)
        if r <= MUTPB:
            chunk[8] = np.random.randint(0, 11)

        ### Mutate bias regularizer.
        #chunk[9] = tools.mutGaussian([ chunk[9] ], chunk[9], .1, MUTPB)
        r = np.random.uniform(0, 1)
        if r <= MUTPB:
            chunk[9] = np.random.normal(chunk[9], .1)

        ### Mutate activity regularizer.
        #chunk[10] = tools.mutGaussian([ chunk[10] ], chunk[10], .1, MUTPB)
        r = np.random.uniform(0, 1)
        if r <= MUTPB:
            chunk[10] = np.random.normal(chunk[10], .1)

        ### Mutate kernel initializer.
        r = np.random.uniform(0, 1)
        if r <= MUTPB:
            chunk[11] = np.random.randint(0, 11)

        ### Mutate dropout rate.
        r = np.random.uniform(0,1)
        if r <= MUTPB:
            chunk[12] = np.random.normal(chunk[12], .1)
        if chunk[12] > 1:
            chunk[12] = .9
        elif chunk[12] < 0:
            chunk[12] = 0

        ### Update the chunk (layer).
        #chunk = [ chunk[0][0][0], chunk[1][0][0], chunk[2][0][0],
        #          chunk[3][0][0], chunk[4][0][0], chunk[5][0][0],
        #          chunk[6][0][0], chunk[7][0][0], chunk[8][0][0],
        #          chunk[9][0][0], chunk[10][0][0] ]
        #chunk = [ chunk[0], chunk[1], chunk[2], 
        #          chunk[3], chunk[4], chunk[5],
        #          chunk[6], chunk[7], chunk[8],
        #          chunk[9], chunk[10] ]

        ### Add chunk (layer) to the mutated individual.
        mutated_individual += chunk

    ### Create the mutated individual.
    mutated_individual = creator.Individual(mutated_individual)

    return mutated_individual

### Check if the x and y kernals are valid (they should
### be smaller than the corresponding dimension size.)

def check_kernel_validity(individual, original_x_dimension, original_y_dimension):
    ### Divide chromosome into chunks for each layer.
    #chunks = divide_chunks(individual, 10)
    chunks = divide_chunks(individual, 13)

    ### Set previous x and y dimensions equal to the original x and y dimensions.
    ### (the x and y dimensions of the data set before the first layer.)
    previous_x_dimension = original_x_dimension
    previous_y_dimension = original_y_dimension

    ### Start a list for the modified individual. (The modified individual has its kernal sizes
    ### modified if the kernal size is invalid. Saves the unmodified individual if the kernal
    ### size is valid.)
    modified_individual = []
    #modified_individual += [ num_of_layers ]

    ### Iterate over the layers.
    for chunk in chunks:
        ### Get the kernal size for the x and y dimensions.
        kernel_x = chunk[3]
        kernel_y = chunk[4]

        ### Check if the kernal size is greater than the dimension size. The kernel size
        ### needs to be less than or equal to the dimension size minus 1 (for stride = 1).
        ### output = input - (kernel - 1).
        ### If the kernal size is too large, generate a random number between 0 and 1 and
        ### take the floor of [that number times the dimension size]. This gives a kernal size
        ### that is less than the dimension size.

        if kernel_x > previous_x_dimension - 1:
            kernel_x_ratio = np.random.uniform(0, 1)
            kernel_x = int(math.floor(kernel_x_ratio * previous_x_dimension))

        if kernel_y > previous_y_dimension - 1:
            kernel_y_ratio = np.random.uniform(0, 1)
            kernel_y = int(math.floor(kernel_y_ratio * previous_y_dimension))

        ### Check if the kernal size is less than 1. If so, change the size to be equal to 1.
        if kernel_x < 1:
            kernel_x = 1
        if kernel_y < 1:
            kernel_y = 1

        ### New dimension size is the old dimension size minus the quantity
        ### kernel size minus 1.

        previous_x_dimension -= (kernel_x - 1)
        previous_y_dimension -= (kernel_y - 1)

        ### Check if the kernel size is less than 1. If so, set the kernel
        ### size to be equal to 1.

        if previous_x_dimension < 1:
            previous_x_dimension = 1
            kernel_x = 1
        if previous_y_dimension < 1:
            previous_y_dimension = 1
            kernel_y = 1

        ### Save the new, valid kernel sizes to the chunk (layer).
        if kernel_x < 1:
            kernel_x = 1
        if kernel_y < 1:
            kernel_y = 1

        chunk[3] = int(math.floor(kernel_x))
        chunk[4] = int(math.floor(kernel_y))

        ### Save modified chunk to the new chromosome.
        modified_individual += chunk

    ### Create the modified individual using keras.creator.Individual.
    modified_individual = creator.Individual(modified_individual)

    return modified_individual

#def num_layers():
#    return np.random.randint(1, max_num_layers+1 )
def use_layer():                        ### Return 0 (skip layer) or 1 (use layer).
    return np.random.randint(0, 2)
def layer():                            ### Return random integer between 0 and 2 for layer type.
    return np.random.randint(6)
def output_dimensionality():            ### Return random integer between 2 and 100 for number of output_dimensionality for layer.
    #return np.random.randint(2, 101)
    #return np.random.randint(2, 11)
    return 2
def get_kernel_x(kernel_x):             ### Return kernel size for x dimension (not sure why I did this).
    return kernel_x
def get_kernel_y(kernel_y):             ### Return kernel size for y dimension (not sure why I did this).
    return kernel_y
def strides():
    return np.random.randint(1,11)
def act():                              ### Return random integer between 0 and 10 for layer activation type.
    return np.random.randint(11)
def use_bias():                         ### Return random integer between 0 and 1 for use_bias = True or False.
    return np.random.randint(2)
def bias_init():                        ### Return random integer between 0 and 10 for bias initializer for layer.
    return np.random.randint(11)
def bias_reg():                         ### Return random float between 0 and 1 for bias regularizer for layer.
    return np.random.uniform()
def act_reg():                          ### Return random float between 0 and 1 for activation regularizer for layer.
    return np.random.uniform()

'''
def seed_population(population, seed_file_name):
    #print(len(population))
    with open(seed_file_name, 'r') as fil:
        seed_individuals = fil.readlines()
    #print(len(seed_individuals))
    for i, seed in enumerate(seed_individuals):
        population[i] = list(seed)
    print(population)
    return population
'''
'''
def seed_population(population, seed_file_name):
    with open(seed_file_name, 'rb') as fil:
        seed_individuals = pickle.load(fil)
    for i, seed in enumerate(seed_individuals):
        population[i] = seed_individuals[i]
    
    #print(population)

    return population
'''

### Extract training data.
with open('../x_train.pkl', 'rb') as pkl_file:
    x_train = pickle.load(pkl_file, encoding = 'latin1')

### Get dimension of data (size of x and y dimensions).
original_x_dimension = x_train.shape[1]
original_y_dimension = x_train.shape[2]
previous_x_dimension = original_x_dimension
previous_y_dimension = original_y_dimension

#print(previous_x_dimension)
#print(previous_y_dimension)

### Not sure what this does besides setting the weights for each objective function.
#creator.create('FitnessMax', base.Fitness, weights=(1., 1., 1., 1., 1.))
creator.create('FitnessMax', base.Fitness, weights=(1., 1.))
creator.create('Individual', list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
### Create a string of a command to be executed to register a 'type' of individual.
### The 'type' of individual depends on how many layers the individual has.
toolbox_ind_str = "toolbox.register('individual', tools.initCycle, creator.Individual, ("

### Iterate over the number of layers and append to string to be executed.
for n in range(max_num_layers):
    use_layer_str = 'use_layer_' + str(n)
    layer_str = 'layer_' + str(n)
    output_dimensionality_str = 'output_dimensionality_' + str(n)
    kernel_x_str = 'kernel_x_' + str(n)
    kernel_y_str = 'kerner_y_' + str(n)
    strides_str = 'strides_' + str(n)
    act_str = 'act_' + str(n)
    use_bias_str = 'use_bias_' + str(n)
    bias_init_str = 'bias_init_' + str(n)
    bias_reg_str = 'bias_reg_' + str(n)
    act_reg_str = 'act_reg_' + str(n)
    
    toolbox.register(use_layer_str, use_layer)
    toolbox.register(layer_str, layer)
    toolbox.register(output_dimensionality_str, output_dimensionality)

    kernel_x_ratio = np.random.uniform(0, 1)
    kernel_x = int(math.floor(kernel_x_ratio * previous_x_dimension))
    kernel_y_ratio = np.random.uniform(0, 1)
    kernel_y = int(math.floor(kernel_y_ratio * previous_y_dimension))

    if kernel_x < 1:
        kernel_x = 1
    if kernel_y < 1:
        kernel_y = 1

    previous_x_dimension -= (kernel_x - 1)
    previous_y_dimension -= (kernel_y - 1)

    if previous_x_dimension < 1:
        previous_x_dimension = 1
        kernel_x = 1
    if previous_y_dimension < 1:
        previous_y_dimension = 1
        kernel_y = 1

    toolbox.register(kernel_x_str, get_kernel_x, kernel_x)
    toolbox.register(kernel_y_str, get_kernel_y, kernel_y)
    toolbox.register(act_str, act)
    toolbox.register(strides_str, strides)
    toolbox.register(use_bias_str, use_bias)
    toolbox.register(bias_init_str, bias_init)
    toolbox.register(bias_reg_str, bias_reg)
    toolbox.register(act_reg_str, act_reg)

    toolbox_ind_str += 'toolbox.' + use_layer_str + ', toolbox.' + layer_str + ', toolbox.' + output_dimensionality_str + ', toolbox.' + kernel_x_str + ', toolbox.' + kernel_y_str + ', toolbox.' + strides_str + ', toolbox.' + act_str + ', toolbox.' + use_bias_str + ', toolbox.' + bias_init_str + ', toolbox.' + bias_reg_str + ', toolbox.' + act_reg_str
    if n != max_num_layers-1:
        toolbox_ind_str += ", "

toolbox_ind_str += "), n=1)"

### Execute string to register individual type.
exec(toolbox_ind_str)

### Register population, mate, mutate, select, and evaluate functions.
toolbox.register('population', tools.initRepeat, list, toolbox.individual)
toolbox.register('mate', tools.cxTwoPoint)
toolbox.register('mutate', myMutation)
#toolbox.register('select', tools.selNSGA2)
toolbox.register('select', tools.selTournament, tournsize = 2)
#toolbox.register('evaluate', evaluate)

def main():
    ### Population size.
    #population_size = 32
    #population_size = 4
    #print('population_size: ', population_size)

    ### Number of individuals (parents) to clone for the next generation.
    #selection_size = 16
    #selection_size = 2
    #print('selection_size: ', selection_size)

    ### Number of individuals made through crossing of selected parents.
    cross_over_size = population_size - selection_size

    print('Setting up population.\n')

    ### Set up population.
    pop = toolbox.population(n=population_size)

    #print('population')
    #for p in pop:
    #    print(p)

    print('population_size: ', population_size)
    print('selection_size: ', selection_size)

    ### Set mutation probability and number of generations.
    #MUTPB = .2
    print('MUTPB: ', MUTPB)
    #NGEN = 100
    #NGEN = 1
    print('NGEN: ', NGEN)

    #pop = seed_population(pop, 'seed_002_use_this.pkl')
    #print('population')
    #for p in pop:
    #    print(p)

    '''

    ### Evaluate fitness of initial population.
    #fitnesses = list( toolbox.map(toolbox.evaluate, pop) )
    fitnesses = [evaluate(ind, i, 0) for i, ind in enumerate(pop)]

    ### Save fitness values to individuals.
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    ### Create directory to save the data for the 0th generation.
    generation_dir_name = data_directory_name + '{0:04d}/generation_00000/'.format(next_dir_number)
    if not os.path.isdir(generation_dir_name):
        os.makedirs(generation_dir_name)

    ### Save the population of the 0th generation.
    generation_population_file_name = generation_dir_name + '{0}{1:02d}{2:02d}_{3:04d}_generation_00000_population.pkl'.format(year, month, day, next_dir_number)
    with open(generation_population_file_name, 'wb') as fil:
        pickle.dump(pop, fil)

    ### Save the fitnesses of the 0th generation.
    generation_fitness_file_name = generation_dir_name + '{0}{1:02d}{2:02d}_{3:04d}_generation_00000_fitness.pkl'.format(year, month, day, next_dir_number)
    with open(generation_fitness_file_name, 'wb') as fil:
        pickle.dump(fitnesses, fil)

    '''

    for p in pop:
        print(p)

    '''

    ### Iterate over the generations.
    for g in range(1, NGEN):
        ### Select the parents.
        selected_parents = toolbox.select( pop, selection_size )

        print('Generation {} ... \n'.format(g))

        new_children = []

        ### Mate the parents to form new individuals.
        for c in range(cross_over_size):
            parent1, parent2 = random.sample(selected_parents, 2)
            parent1, parent2 = toolbox.clone(parent1), toolbox.clone(parent2)
            child1, child2 = toolbox.mate(parent1, parent2)
            del child1.fitness.values
            new_children.append(child1)
            ### Why is only one of the children saved? I think it works out
            ### this way so that there's not too many individuals.
            ### This could probably change in the future.

        ### New population consists of selected parents and their children.
        new_population = selected_parents + new_children

        ### Mutate the new population.
        for m, mutant in enumerate(new_population):
            r = np.random.uniform(0, 1)
            if r <= MUTPB:
                mutated_ind = toolbox.mutate(mutant)
                del mutated_ind.fitness.values
                new_population[m] = mutated_ind

        ### Check the kernel validity of the new population.
        new_population = [check_kernel_validity(ind, original_x_dimension, original_y_dimension) for ind in new_population]

        #fitnesses = list( map(toolbox.evaluate, new_population) )
        fitnesses = [evaluate(ind, i, g) for i, ind in enumerate(pop)]

        for ind, fit in zip(new_population, fitnesses):
            ind.fitness.values = fit

        ### Set pop to be the new population.
        pop = new_population

        ### Create directory to save the data for the g-th generation.
        generation_dir_name = data_directory_name + '{0:04d}/generation_{1:05d}/'.format(next_dir_number, g)
        if not os.path.isdir(generation_dir_name):
            os.makedirs(generation_dir_name)

        ### Save the population of the g-th generation.
        generation_population_file_name = generation_dir_name + '{0}{1:02d}{2:02d}_{3:04d}_generation_{4:05d}_population.pkl'.format(year, month, day, next_dir_number, g)
        with open(generation_population_file_name, 'wb') as fil:
            pickle.dump(pop, fil)

        ### Save the fitnesses of the g-th generation.
        generation_fitness_file_name = generation_dir_name + '{0}{1:02d}{2:02d}_{3:04d}_generation_{4:05d}_fitness.pkl'.format(year, month, day, next_dir_number, g)
        with open(generation_fitness_file_name, 'wb') as fil:
            pickle.dump(fitnesses, fil)

    '''

    ### Return the final population and final fitnesses.
    #return pop, fitnesses

main()
#pop, fitnesses = main()
