
import os
#import subprocess


#for i in range(24):
    #os.system('srun --ntasks 1 --nodes 1 --nodelist=node004 python3 AlexNet.py &> AlexNet_runs_max_300/AlexNet_max_300_' + str(i) + '.txt &')
    #os.system('srun --ntasks 1 --nodes 1 --nodelist=node005 python3 LeNet.py &> LeNet_runs_max_300/LeNet_max_300_' + str(i) + '.txt &')
    #os.system('srun --ntasks 1 --nodes 1 --nodelist=node004 python3 VGG_baseline_01_VGG_block.py &> VGG_baseline_01_VGG_block_runs_max_300/VGG_baseline_01_VGG_block_max_300_' + str(i) + '.txt &')
    #os.system('srun --ntasks 1 --nodes 1 --nodelist=node005 python3 VGG_baseline_02_VGG_block.py &> VGG_baseline_02_VGG_block_runs_max_300/VGG_baseline_02_VGG_block_max_300_' + str(i) + '.txt &')
    #os.system('srun --ntasks 1 --nodes 1 --nodelist=node006 python3 VGG_baseline_03_VGG_block.py &> VGG_baseline_03_VGG_block_runs_max_300/VGG_baseline_03_VGG_block_max_300_' + str(i) + '.txt &')
    #os.system('srun --ntasks 1 --nodes 1 --nodelist=node007 python3 VGG_baseline_03_VGG_block_dropout.py &> VGG_baseline_03_VGG_block_dropout_runs_max_300/VGG_baseline_03_VGG_block_dropout_max_300_' + str(i) + '.txt &')
    #os.system('srun --ntasks 1 --nodes 1 --nodelist=node008 python3 VGG_baseline_03_VGG_block_dropout_batch_normalization.py &> VGG_baseline_03_VGG_block_dropout_batch_normalization_runs_max_300/VGG_baseline_03_VGG_block_dropout_batch_normalization_max_300_' + str(i) + '.txt &')


for i in range(24):
    print('Running job {} for VGG_baseline_01'.format(i))
    run_file_name = '/exports/home/j_liu21/projects/genetic_algorithms/optimizing_neural_nets/VGG/VGG_baseline_01_VGG_block.py'
    #run_file_name = '/exports/home/j_liu21/projects/genetic_algorithms/optimizing_neural_nets/VGG/VGG_baseline_02_VGG_block.py'
    #run_file_name = '/exports/home/j_liu21/projects/genetic_algorithms/optimizing_neural_nets/VGG/VGG_baseline_03_VGG_block_dropout_batch_normalization.py'
    #run_file_name = '/exports/home/j_liu21/projects/genetic_algorithms/optimizing_neural_nets/VGG/VGG_baseline_03_VGG_block_dropout.py'
    #run_file_name = '/exports/home/j_liu21/projects/genetic_algorithms/optimizing_neural_nets/VGG/VGG_baseline_03_VGG_block.py'

    save_file_name = 'VGG/VGG_baseline_01_VGG_block_run_003/VGG_baseline_01_VGG_block_run_004_{0:03d}.txt'.format(i)
    #save_file_name = 'VGG/VGG_baseline_02_VGG_block_run_003/VGG_baseline_02_VGG_block_run_003_{0:03d}.txt'.format(i)
    #save_file_name = 'VGG/VGG_baseline_03_VGG_block_dropout_batch_normalization_run_003/VGG_baseline_03_VGG_block_dropout_batch_normalization_run_003_{0:03d}.txt'.format(i)
    #save_file_name = 'VGG/VGG_baseline_03_VGG_block_dropout_run_003/VGG_baseline_03_VGG_block_run_003_{0:03d}.txt'.format(i)
    #save_file_name = 'VGG/VGG_baseline_03_VGG_block_run_003/VGG_baseline_03_VGG_block_run_003_{0:03d}.txt'.format(i)

    #proc = subprocess.Popen(['srun', '--ntasks', '1', '--nodes', '1', '--nodelist=node002', 'python3.6', run_file_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    #proc = subprocess.Popen(['srun', '--ntasks', '1', '--nodes', '1', '--nodelist=node002', 'python3.6', run_file_name, '&>', save_file_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    #proc = subprocess.Popen(['srun', '--ntasks', '1', '--nodes', '1', '--nodelist=node002', 'python3.6', run_file_name, '&>', save_file_name])

    os.system('srun --ntasks 1 --nodes 1 --cpus-per-task 48 --nodelist=node005 python3.6 ' + run_file_name + ' &> ' + save_file_name )
    #out = proc.communicate()[0].decode('utf-8')
    #print(out)
