
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import sys
sys.path.insert(1, '/exports/home/j_liu21/projects/genetic_algorithms/optimizing_neural_nets/functions/')

from divide_chunks import divide_chunks

from deap import base
from deap import creator

creator.create('FitnessMax', base.Fitness, weights=(1., 1., 1., 1., 1.))
creator.create('Individual', list, fitness=creator.FitnessMax)


num_layers_list = []
num_layer_type_list_1 = []
num_layer_type_list_2 = []
num_layer_type_list_3 = []
num_layer_type_list_4 =	[]
num_layer_type_list_5 = []
num_layer_type_list_6 =	[]
num_layer_type_list = [ num_layer_type_list_1, num_layer_type_list_2, num_layer_type_list_3,
                        num_layer_type_list_4, num_layer_type_list_5, num_layer_type_list_6 ]

first_layer = []
second_layer = []
third_layer = []
fourth_layer = []
fifth_layer = []
sixth_layer = []
freq_layer = [ first_layer, second_layer, third_layer, fourth_layer, fifth_layer, sixth_layer ]


parent_dir_name = '/exports/home/j_liu21/projects/genetic_algorithms/optimizing_neural_nets/data/2020/202001/20200130/0009/'


for dir in sorted(os.listdir(parent_dir_name)[-100:]):
    #print(dir)
    generation_dir_name = parent_dir_name + dir + '/'
    #print(generation_dir_name)
    for fil in os.listdir(generation_dir_name):
        if 'population' in fil:
            population_file_name = generation_dir_name + fil
    #print(population_file_name)
    with open(population_file_name, 'rb') as pkl:
        population = pickle.load(pkl)

    #num_layers = 0

    #print(population)
    for p in population:
        num_layers = 0
        chunks = divide_chunks(p, 11)
        for chunk in chunks:
            if chunk[0] == 1:
                num_layers += 1
                if num_layers < 7:
                    freq_layer[num_layers-1].append(chunk[1])

        #print(p)
        #print('\n')
        #if p[0] == 1:
        #    num_layers += 1

        num_layers_list.append(num_layers)

        if num_layers >= 1 and num_layers <= 6:
            for chunk in chunks:
                if chunk[0] == 1:
                    num_layer_type_list[num_layers-1].append(chunk[1])

    #if num_layers == 1:
    #    for p in population:
    #        if p[0] == 1:
    #            num_layer_type_list.append(p[1])

#print(num_layer_type_list[0])

'''
Dense = []
Conv2D = []
SeparableConv2D = []
DepthwiseConv2D = []
MaxPooling2D = []
AveragePooling2D = []

for c, n in enumerate(num_layer_type_list):
    for x in n:
        if x == 1:
            Dense.append(c+1)
        elif x == 2:
            Conv2D.append(c+1)
        elif x == 3:
            SeparableConv2D.append(c+1)
        elif x == 4:
            DepthwiseConv2D.append(c+1)
        elif x == 5:
            MaxPooling2D.append(c+1)
        elif x == 6:
            AveragePooling2D.append(c+1)
'''
'''
labels, counts = np.unique(num_layers_list, return_counts=True)
plt.bar(labels, counts, align='center')
plt.gca().set_xticks(labels)
plt.grid('--')
plt.title('Prevalence of different number of layers')
plt.xlabel('Number of layers')
plt.ylabel('Number of Individuals')

plt.show()
'''

labels_1, counts_1 = np.unique(num_layer_type_list[0], return_counts=True)
plt.bar(labels_1-.3, counts_1, width=.1, align='center', label='Dense')

labels_2, counts_2 = np.unique(num_layer_type_list[1], return_counts=True)
plt.bar(labels_2-.2, counts_2, width=.1, align='center', label='Conv2D')

labels_3, counts_3 = np.unique(num_layer_type_list[2], return_counts=True)
plt.bar(labels_3-.1, counts_3, width=.1, align='center', label='SeparableConv2D')

labels_4, counts_4 = np.unique(num_layer_type_list[3], return_counts=True)
plt.bar(labels_4, counts_4, width=.1, align='center', label='DepthwiseConv2D')

labels_5, counts_5 = np.unique(num_layer_type_list[4], return_counts=True)
plt.bar(labels_5+.1, counts_5, width=.1, align='center', label='MaxPooling2D')

labels_6, counts_6 = np.unique(num_layer_type_list[5], return_counts=True)
plt.bar(labels_6+.2, counts_6, width=.1, align='center', label='AveragePooling')

plt.grid('--')
plt.title('Prevalence of layer type in networks with different number of hidden layers')
#plt.xticks(rotation=45)
locs, labels = plt.xticks()
labels = ['1', '2', '3', '4', '5', '6']
plt.xticks(locs[:-1]+1, labels, horizontalalignment='center')

plt.xlabel('Number of Layers')
plt.ylabel('Number of Each Layer Type')
plt.legend()

plt.show()

labels = [  "Dense",
            "Conv2D",
            "SeparableConv2D",
            "DepthwiseConv2D",
            "MaxPooling2D",
            "AveragePooling2D" ]

'''
fig, axes = plt.subplots(2, 3)

#plt.setp(axes.xaxis.get_majorticklabels(), rotation=45)

labels_1, counts_1 = np.unique(freq_layer[0], return_counts=True)
f1 = axes[0,0].bar(labels_1, counts_1, align='center')
axes[0,0].title.set_text('First Layer')
axes[0,0].grid('--')

#print(labels_1)
#print(counts_1)

labels_2, counts_2 = np.unique(freq_layer[1], return_counts=True)
f2 = axes[0,1].bar(labels_2, counts_2, align='center')
axes[0,1].title.set_text('Second Layer')
axes[0,1].grid('--')

labels_3, counts_3 = np.unique(freq_layer[2], return_counts=True)
f3 = axes[0,2].bar(labels_3, counts_3, align='center')
axes[0,2].title.set_text('Third Layer')
axes[0,2].grid('--')

labels_4, counts_4 = np.unique(freq_layer[3], return_counts=True)
f4 = axes[1,0].bar(labels_4, counts_4, align='center')
axes[1,0].title.set_text('Fourth Layer')
axes[1,0].grid('--')

labels_5, counts_5 = np.unique(freq_layer[4], return_counts=True)
f5 = axes[1,1].bar(labels_5, counts_5, align='center')
axes[1,1].title.set_text('Fifth Layer')
axes[1,1].grid('--')

labels_6, counts_6 = np.unique(freq_layer[5], return_counts=True)
f6 = axes[1,2].bar(labels_6, counts_6, align='center')
axes[1,2].title.set_text('Sixth Layer')
axes[1,2].grid('--')

ff = [f1, f2, f3, f4, f5, f6]
for f in ff:
    f[0].set_color('blue')
    f[1].set_color('orange')
    f[2].set_color('green')
    f[3].set_color('red')
    f[4].set_color('purple')
    f[5].set_color('brown')


dense = mpatches.Patch(color='blue', label='0: Dense')
conv2d = mpatches.Patch(color='orange', label='1: Conv2D')
sepconv2d = mpatches.Patch(color='green', label='2: SeparableConv2D')
depthconv2d = mpatches.Patch(color='red', label='3: DepthwiseConv2D')
maxpool2d = mpatches.Patch(color='purple', label='4: MaxPooling2D')
avgpool2d = mpatches.Patch(color='brown', label='5: AveragePooling2D')
plt.legend(handles=[dense, conv2d, sepconv2d, depthconv2d, maxpool2d, avgpool2d])

plt.show()
'''
