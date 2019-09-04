#!/usr/bin/python3

import random
import pickle
import numpy as np
import subprocess
import math
import time

from deap import base
from deap import creator
from deap import tools

import genotype_to_phenotype

### For individuals with multiple layers, this function chops up the vector for an individual into chunks for each layer.
### Don't know how this function works. Got this from the internet.
def divide_chunks(my_list, n):
    chunks = [my_list[i * n:(i + 1) * n] for i in range((len(my_list) + n - 1) // n )]
    return chunks

### Top and bottom wrapper text to be used in making the neural network.
top_file = 'wrapper_text/top_text.txt'
bot_file = 'wrapper_text/bot_text.txt'

with open(top_file, 'r') as top:
    top_lines = top.readlines()
with open(bot_file, 'r') as bot:
    bot_lines = bot.readlines()

### Get the fitness of an individual.
def evaluate(individual):
    x_chunks = divide_chunks(individual, 9)

    ### Convert genotype to phenotype.
    phenotype_list = []
    for chunk in x_chunks:
        phenotype = genotype_to_phenotype.get_phenotype(chunk)
        phenotype_list.append(phenotype)

    ### Open temporary file.
    temp_file_name = 'temp_file.py'

    with open(temp_file_name, 'w') as tempfile:
        ### Write top wrapper to file.
        for line in top_lines:
            tempfile.write( line.split('\n')[0] + '\n')

        ### Randomly choose the number of nodes in the first layer.
        number_of_first_layer_nodes = np.random.randint(2, 100)

        ### Write first layer to file.
        tempfile.write("model.add(Conv2D(" + str(number_of_first_layer_nodes) + ", (3, 3), padding='same', activation='relu', input_shape=x_train.shape[1:]))\n")

        ### Write phenotype to file.
        for phenotype in phenotype_list:
            tempfile.write(phenotype + '\n')

        ### Write bottom wrapper to file.
        for line in bot_lines:
            tempfile.write( line.split('\n')[0] + '\n' )

    ### Save time at which job was started.
    start = time.time()

    ### Start the job.
    proc = subprocess.Popen(['python3', temp_file_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    ### Save time at which job ended.
    end = time.time()

    ### Compute the runtime of the job.
    duration = end-start

    ### Compute the duration fitness based on how long it took to run the job.
    inverse_duration = 1./duration

    ### Capture the output of the job.
    out = proc.communicate()[0]

    #print(out)

    ### Compute loss, memory, and cpu usage fitnesses.
    inverse_loss = 1./float(out.upper().split()[-7])
    inverse_mem = 1./float(out.upper().split()[-3])
    inverse_cpu = 1./float(out.upper().split()[-1])

    ### Save the accuracy.
    accuracy = float(out.upper().split()[-5])

    ### Collect the fitness values.
    fitness = ( accuracy, inverse_loss, inverse_duration, inverse_mem, inverse_cpu )
    # fitness = float(out.upper().split()[-1]),
    #print fitness

    ### Return the fitness values.
    return fitness

### Define how offspring mutate.
def myMutation(individual):
    #print('individual: ', individual)
    chunks = divide_chunks(individual, 9)
    #print('chunks:', chunks)
    ### Start a list for mutated individuals.
    mutated_individual = []

    ### Iterate over individuals.
    #print(chunks)
    for chunk in chunks:
        ### Mutate layer type.
        chunk[0] = tools.mutUniformInt([ chunk[0] ], 0, 2, .5)

        ### Mutate kernel x number. (This is expressed as a fraction of the dimension length.)
        chunk[1] = tools.mutGaussian([ chunk[1] ], chunk[1], .1, .5)

        ### Mutate kernel y number. (This is expressed as a fraction of the dimension length.)
        chunk[2] = tools.mutGaussian([ chunk[2] ], chunk[2], .1, .5)

        ### Make the ratios positive.
        if chunk[1][0][0] < 0:
            chunk[1][0][0] += 1
        if chunk[2][0][0] < 0:
            chunk[2][0][0] += 1

        ### Mutate number of nodes.
        chunk[3] = tools.mutUniformInt([ chunk[3] ], 2, 4, .5)

        ### Mutate activation type.
        chunk[4] = tools.mutUniformInt([ chunk[4] ], 0, 10, .5)
 
        ### Mutate use bias.
        chunk[5] = tools.mutUniformInt([ chunk[5] ], 0, 1, .5)

        ### Mutate bias initializer.
        chunk[6] = tools.mutUniformInt([ chunk[6] ], 0, 10, .5)

        ### Mutate bias regularizer.
        chunk[7] = tools.mutGaussian([ chunk[7] ], chunk[7], .1, .5)

        ### Mutate activity regularizer.
        chunk[8] = tools.mutGaussian([ chunk[8] ], chunk[8], .1, .5)

        ### Update the chunk (layer).
        chunk = [ chunk[0][0][0], chunk[1][0][0], chunk[2][0][0],
                  chunk[3][0][0], chunk[4][0][0], chunk[5][0][0],
                  chunk[6][0][0], chunk[7][0][0], chunk[8][0][0] ]

        ### Add chunk (layer) to the mutated individual.
        mutated_individual += chunk

    ### Create the mutated individual.
    mutated_individual = creator.Individual(mutated_individual)

    return mutated_individual

def check_kernel_validity(individual, orginal_x_dimension, original_y_dimension):
    chunks = divide_chunks(individual, 9)
    #print('chunks: ', chunks)
    previous_x_dimension = original_x_dimension
    previous_y_dimension = original_y_dimension
    modified_individual = []
    for chunk in chunks:
        kernel_x = chunk[2]
        kernel_y = chunk[3]

        if kernel_x > previous_x_dimension - 1:
            kernel_x_ratio = np.random.uniform(0, 1)
            kernel_x = int(math.floor(kernel_x_ratio * previous_x_dimension))
        if kernel_y > previous_y_dimension - 1:
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

        chunk[2] = kernel_x
        chunk[3] = kernel_y

        modified_individual += chunk

    modified_individual = creator.Individual(modified_individual)

    return modified_individual

def layer():
    return np.random.randint(3)
def nodes():
    return np.random.randint(2, 100)
def get_kernel_x(kernel_x):
    return kernel_x
def get_kernel_y(kernel_y):
    return kernel_y
def act():
    return np.random.randint(11)
def use_bias():
    return np.random.randint(2)
def bias_init():
    return np.random.randint(11)
def bias_reg():
    return np.random.uniform()
def act_reg():
    return np.random.uniform()

with open('../x_train.pkl', 'rb') as pkl_file:
    x_train = pickle.load(pkl_file, encoding = 'latin1')

original_x_dimension = x_train.shape[1]
original_y_dimension = x_train.shape[2]
previous_x_dimension = original_x_dimension
previous_y_dimension = original_y_dimension

#creator.create('FitnessMax', base.Fitness, weights=(1, 1, 1))
creator.create('FitnessMax', base.Fitness, weights=(1., 1., 1., 1., 1.))
creator.create('Individual', list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox_ind_str = "toolbox.register('individual', tools.initCycle, creator.Individual, ("

#num_layers = 10
num_layers = 1

for n in range(num_layers):
    layer_str = 'layer_' + str(n)
    nodes_str = 'nodes_' + str(n)
    kernel_x_str = 'kernel_x_' + str(n)
    kernel_y_str = 'kerner_y_' + str(n)
    act_str = 'act_' + str(n)
    use_bias_str = 'use_bias_' + str(n)
    bias_init_str = 'bias_init_' + str(n)
    bias_reg_str = 'bias_reg_' + str(n)
    act_reg_str = 'act_reg_' + str(n)

    toolbox.register(layer_str, layer)
    toolbox.register(nodes_str, nodes)

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
    toolbox.register(use_bias_str, use_bias)
    toolbox.register(bias_init_str, bias_init)
    toolbox.register(bias_reg_str, bias_reg)
    toolbox.register(act_reg_str, act_reg)

    toolbox_ind_str += 'toolbox.' + layer_str + ', toolbox.' + nodes_str + ', toolbox.' + kernel_x_str + ', toolbox.' + kernel_y_str + ', toolbox.' + act_str + ', toolbox.' + use_bias_str + ', toolbox.' + bias_init_str + ', toolbox.' + bias_reg_str + ', toolbox.' + act_reg_str
    if n != num_layers-1:
        toolbox_ind_str += ", "

toolbox_ind_str += "), n=1)"

exec(toolbox_ind_str)

toolbox.register('population', tools.initRepeat, list, toolbox.individual)

toolbox.register('mate', tools.cxTwoPoint)
toolbox.register('mutate', myMutation)
toolbox.register('select', tools.selNSGA2)
toolbox.register('evaluate', evaluate)

population_list = []
fitness_list = []

def main():
    #pop = toolbox.population(n=32)
    pop = toolbox.population(n=2)
    print('generation_-1_population: {}\n'.format(pop) )
    #population_list.append( pop )
    #CXPB, MUTPB, NGEN = .5, .5, 100
    CXPB, MUTPB, NGEN = .5, .5, 1

    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    #print('invalid_ind: ', invalid_ind)
    fitnesses = list( toolbox.map(toolbox.evaluate, invalid_ind) )

    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    #pop = toolbox.select(pop, len(pop))

    for g in range(NGEN):
        #offspring = tools.selTournamentDCD(pop, len(pop))
        offspring = toolbox.select(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]

        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            if random.random() <= CXPB:
                toolbox.mate(ind1, ind2)

            toolbox.mutate(ind1)
            toolbox.mutate(ind2)
            del ind1.fitness.values, ind2.fitness.values

        for i, inv_ind in enumerate(invalid_ind):
            inv_ind = check_kernel_validity(inv_ind, original_x_dimension, original_y_dimension)
            invalid_ind[i] = inv_ind

        fitnesses = list( toolbox.map(toolbox.evaluate, invalid_ind) )

        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        fitnesses = list( toolbox.map(toolbox.evaluate, offspring) )

        for off, fit in zip(offspring, fitnesses):
            off.fitness.values = fit

        fitnesses = list( map(toolbox.evaluate, pop) )
        print('generation_{}_fitnesses: {}'.format(g, fitnesses) )
        #fitness_list.append( fitnesses )

        pop = toolbox.select(pop + offspring, len(pop))

        pop[:] = offspring
        print('generation_{}_population: {}'.format(g, pop) )
        #population_list.append( pop )

    fitnesses = list( map(toolbox.evaluate, pop) )

    # fits = [ind.fitness.values[0] for ind in pop]

    return pop, fitnesses
    #return population_list, fitness_list

# main()
pop, fitnesses = main()
#population_list, fitness_list = main()
#data = [population_list, fitness_list]

#with open('20190829_test_01.pkl', 'wb') as fil:
#    pickle.dump(data, fil, protocol = 2)

#print('final_population: {}'.format(pop) )
#print('final_fitnesses: {}'.format(fitnesses) )


# print original_x_dimension, '\n'
# print original_y_dimension, '\n'
# x = [2, 3, 22, 27, 0, 1, 6, 0.19334968375277173, 0.06346641132347908, 0, 3, 30, 30, 3, 1, 0, 0.496359412700602, 0.5837478239730752, 1, 2, 30, 30, 9, 0, 1, 0.6322657300752614, 0.7064246891981825, 2, 2, 30, 30, 10, 1, 7, 0.8706446388657239, 0.6199682951025327]
# print 'original: ', x
# print 'checked: ', check_kernel_validity(x, original_x_dimension, original_y_dimension)
