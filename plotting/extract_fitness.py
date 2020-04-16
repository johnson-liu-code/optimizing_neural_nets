import pickle


#filename = '/exports/home/j_liu21/projects/genetic_algorithms/optimizing_neural_nets/data/2019/201912/20191204/0004/generation_01235/20191204_0004_generation_01235_fitness.pkl'
filename = '/exports/home/j_liu21/projects/genetic_algorithms/optimizing_neural_nets/data/2020/202001/20200130/0009/generation_00201/20200130_0009_generation_00201_fitness.pkl'

with open(filename, 'rb') as fil:
    fitnesses = pickle.load(fil)

#print(fitnesses)
for i, fitness in enumerate(fitnesses):
    if fitness[0] > .4:
        print(i, fitness)
