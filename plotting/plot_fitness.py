import os
import pickle
import numpy as np
from running_average import running_avg4
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

dir_name = '/exports/home/j_liu21/projects/genetic_algorithms/optimizing_neural_nets/data/2020/202001/20200130/0009/'
#dir_name = '/exports/home/j_liu21/projects/genetic_algorithms/optimizing_neural_nets/data/2020/202002/20200204/0008/'
#dir_name = '/exports/home/j_liu21/projects/genetic_algorithms/optimizing_neural_nets/data/2019/201912/20191204/0005/'

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
    #accuracy, inverse_loss = zip(*fitnesses)

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

#accuracy_vs_gen = running_avg4(max_accuracy_vs_gen, 200)
#inverse_loss_vs_gen = running_avg4(max_inverse_loss_vs_gen, 200)
#inverse_duration_vs_gen = running_avg4(max_inverse_duration_vs_gen, 200)
#inverse_mem_vs_gen = running_avg4(max_inverse_mem_vs_gen, 200)
#inverse_cpu_vs_gen = running_avg4(max_inverse_cpu_vs_gen, 200)

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
plt.plot(max_accuracy_vs_gen, label = 'accuracy', linewidth = 1)
#plt.plot(accuracy_vs_gen)
plt.plot(max_inverse_loss_vs_gen, label = 'inverse loss (normalized)', linewidth = 1)
#plt.plot(inverse_loss_vs_gen)
plt.plot(max_inverse_duration_vs_gen, label = 'inverse duration (normalized)', linewidth = 1)
#plt.plot(inverse_duration_vs_gen)
plt.plot(max_inverse_mem_vs_gen, label = 'inverse memory (normalized)', linewidth = 1)
#plt.plot(inverse_mem_vs_gen)
plt.plot(max_inverse_cpu_vs_gen, label = 'inverse cpu usage (normalized)', linewidth = 1)
#plt.plot(inverse_cpu_vs_gen)

plt.grid(linestyle = '--')

plt.xlabel('Generation')
plt.ylabel('Score')
plt.legend(loc = 'lower right')

plt.show()
#plt.savefig('test')

