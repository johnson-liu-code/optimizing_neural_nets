

def extract_ind(file_name):
    with open(file_name, 'r') as fil:
        lines = fil.readlines()

    lines = [line.strip() for line in lines]

    print(lines)

    #lst = [float(i) if '.' in i else int(i) for i in lines.split('\n')]
    #print(lst)


if __name__ == '__main__':
    file_name = '/exports/home/j_liu21/projects/genetic_algorithms/optimizing_neural_nets/initial_population/set_001/ind_001_VGG_baseline_03_VGG_block_variable_dropout.txt'
    extract_ind(file_name)
