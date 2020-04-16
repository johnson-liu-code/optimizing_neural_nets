
import os

for i in range(24):
    #os.system('srun --ntasks 1 --nodes 1 --nodelist=node004 python3 AlexNet.py &> AlexNet_runs_max_300/AlexNet_max_300_' + str(i) + '.txt &')
    os.system('srun --ntasks 1 --nodes 1 --nodelist=node005 python3 LeNet.py &> LeNet_runs_max_300/LeNet_max_300_' + str(i) + '.txt &')
    #os.system('srun --ntasks 1 --nodes 1 --nodelist=node004 python3 VGG_baseline_01_VGG_block.py &> VGG_baseline_01_VGG_block_runs_max_300/VGG_baseline_01_VGG_block_max_300_' + str(i) + '.txt &')
    #os.system('srun --ntasks 1 --nodes 1 --nodelist=node005 python3 VGG_baseline_02_VGG_block.py &> VGG_baseline_02_VGG_block_runs_max_300/VGG_baseline_02_VGG_block_max_300_' + str(i) + '.txt &')
    #os.system('srun --ntasks 1 --nodes 1 --nodelist=node006 python3 VGG_baseline_03_VGG_block.py &> VGG_baseline_03_VGG_block_runs_max_300/VGG_baseline_03_VGG_block_max_300_' + str(i) + '.txt &')
    #os.system('srun --ntasks 1 --nodes 1 --nodelist=node007 python3 VGG_baseline_03_VGG_block_dropout.py &> VGG_baseline_03_VGG_block_dropout_runs_max_300/VGG_baseline_03_VGG_block_dropout_max_300_' + str(i) + '.txt &')
    #os.system('srun --ntasks 1 --nodes 1 --nodelist=node008 python3 VGG_baseline_03_VGG_block_dropout_batch_normalization.py &> VGG_baseline_03_VGG_block_dropout_batch_normalization_runs_max_300/VGG_baseline_03_VGG_block_dropout_batch_normalization_max_300_' + str(i) + '.txt &')

