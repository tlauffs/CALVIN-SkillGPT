# datapath_dd_training = '/media/tim/6f37312f-8eb4-400c-a4e7-e229c18bbf2c/datasets/calvin_debug_dataset/training'

'''
datapath_training = '/home/tim/calvin_debug_dataset/training'
datapath_val = '/home/tim/calvin_debug_dataset/validation'
datapath_training_parsed = '/home/tim/calvin_debug_dataset_parsed/training'
datapath_val_parsed = '/home/tim/calvin_debug_dataset_parsed/validation'
'''

datapath_training = '/media/tim/E/datasets/task_D_D/training'
datapath_val = '/media/tim/E/datasets/task_D_D/validation'
datapath_training_parsed = '/media/tim/E/datasets/task_D_D_parsed/training'
datapath_val_parsed = '/media/tim/E/datasets/task_D_D_parsed/validation'


max_seq_length = 16
batch_size = 32
num_workers = 10

d_model = 2050
dropout = 0.1
n_heads = 2
# dim_feedforward
d_ff = 512
n_layers = 3
epochs = 1000