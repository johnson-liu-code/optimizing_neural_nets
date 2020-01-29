import os
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



dir_name = '/exports/home/j_liu21/projects/genetic_algorithms/optimizing_neural_nets/data/2019/201912/20191210/0005/'
generation_directories = os.listdir(dir_name)

max_accuracy_vs_gen = []
max_inverse_loss_vs_gen =[]
max_inverse_duration_vs_gen = []
max_inverse_mem_vs_gen = []
max_inverse_cpu_vs_gen = []

for generation in sorted(generation_directories):
    generation_directory = dir_name + generation + '/'
    #print(generation_directory)

    generation_files = sorted( os.listdir(generation_directory) )
    fitness_file = generation_directory + generation_files[0]
    with open(fitness_file, 'rb') as fit:
        fitnesses = pickle.load(fit)
    accuracy, inverse_loss, inverse_duration, inverse_mem, inverse_cpu = zip(*fitnesses)
    #inverse_loss = 1./np.array(loss)

    max_accuracy_vs_gen.append( max(accuracy) )
    max_inverse_loss_vs_gen.append( max(inverse_loss) )
    max_inverse_duration_vs_gen.append( max(inverse_duration) )
    max_inverse_mem_vs_gen.append( max(inverse_mem) )
    max_inverse_cpu_vs_gen.append( max(inverse_cpu) )

#print(max_accuracy_vs_gen)

#max_accuracy_vs_gen = np.array(max_accuracy_vs_gen)/max(max_accuracy_vs_gen)
max_inverse_loss_vs_gen = np.array(max_inverse_loss_vs_gen)/max(max_inverse_loss_vs_gen)
max_inverse_duration_vs_gen = np.array(max_inverse_duration_vs_gen)/max(max_inverse_duration_vs_gen)
max_inverse_mem_vs_gen = np.array(max_inverse_mem_vs_gen)/max(max_inverse_mem_vs_gen)
max_inverse_cpu_vs_gen = np.array(max_inverse_cpu_vs_gen)/max(max_inverse_cpu_vs_gen)


#print(max_accuracy_vs_gen)

#print('inverse loss:')
#print(max_inverse_loss_vs_gen)
#for loss in max_inverse_loss_vs_gen:
#    print(loss)
#print('\n')
#print('inverse duration:')
#print(max_inverse_duration_vs_gen)
#for dur in max_inverse_duration_vs_gen:
#    print(dur)
#print('\n')
#print('inverse memory:')
#print(max_inverse_mem_vs_gen)
#for mem in max_inverse_mem_vs_gen:
#    print(mem)
#print('\n')
#print('inverse cpu:')
#print(max_inverse_cpu_vs_gen)
#for cpu in max_inverse_cpu_vs_gen:
#    print(cpu)

'''
with open('acc', 'w') as acc:
    acc.write(max_accuracy_vs_gen)
with open('loss', 'w') as loss:
    loss.write(max_inverse_loss_vs_gen)
with open('duration', 'w') as dur:
    dur.write(max_inverse_duration_vs_gen)
with open('mem', 'w') as mem:
    mem.write(max_inverse_mem_vs_gen)
with open('cpu', 'w') as cpu:
    cpu.write(max_inverse_cpu_vs_gen)
'''

#fig, ax = plt.subplots(3, 2)
plt.plot(max_accuracy_vs_gen, label = 'accuracy')
plt.plot(max_inverse_loss_vs_gen, label = 'inverse loss')
plt.plot(max_inverse_duration_vs_gen, label = 'inverse duration')
plt.plot(max_inverse_mem_vs_gen, label = 'inverse memory')
plt.plot(max_inverse_cpu_vs_gen, label = 'inverse cpu usage')

plt.grid(linestyle = '--')

plt.xlabel('Generation')
plt.ylabel('Normalized Score')
plt.legend(loc = 'lower left')

#plt.show()
plt.savefig('test')

