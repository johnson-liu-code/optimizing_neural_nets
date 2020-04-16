
import sys
sys.path.append('/exports/home/j_liu21/projects/genetic_algorithms/optimizing_neural_nets/functions')
from divide_chunks import divide_chunks

from deap import base
from deap import creator


import pickle


creator.create('FitnessMax', base.Fitness, weights=(1., 1., 1., 1., 1.))
creator.create('Individual', list, fitness=creator.FitnessMax)


extract_indices = [1, 4, 6, 10, 15, 23, 26, 27, 30]
#extract_indices = [6, 12, 27]

#filename = '/exports/home/j_liu21/projects/genetic_algorithms/optimizing_neural_nets/data/2019/201912/20191204/0004/generation_01235/20191204_0004_generation_01235_population.pkl'
filename = '/exports/home/j_liu21/projects/genetic_algorithms/optimizing_neural_nets/data/2020/202001/20200130/0009/generation_00201/20200130_0009_generation_00201_population.pkl'

with open(filename, 'rb') as fil:
    population = pickle.load(fil)

new_individual_list = []

for i in extract_indices:
    #print(population[i])
    #modified_ind = []
    #chunks = divide_chunks(population[i], 10)
    #for chunk in chunks:
        #modified_ind = modified_ind + [1] + chunk
    #print(modified_ind)
    #new_individual_list.append(modified_ind)
    new_individual_list.append(population[i])

print(new_individual_list)
for n in new_individual_list:
    print(n)
    print('\n')

#with open('seed_002_use_this.pkl', 'wb') as fil:
#    pickle.dump(new_individual_list, fil)
