import torch


'''
datapath_training = '/home/tim/calvin_debug_dataset/training'
datapath_val = '/home/tim/calvin_debug_dataset/validation'
datapath_training_parsed = '/home/tim/calvin_debug_dataset_parsed/training'
datapath_val_parsed = '/home/tim/calvin_debug_dataset_parsed/validation'
'''

datapath_training = '/media/tim/E/datasets/task_D_D/training'
datapath_val = '/media/tim/E/datasets/task_D_D/validation'
#datapath_training_parsed = '/media/tim/E/datasets/task_D_D_parsed/training'
#datapath_val_parsed = '/media/tim/E/datasets/task_D_D_parsed/validation'

datapath_training_parsed = '/media/tim/E/datasets/task_D_D_parsed_combined/training'
datapath_val_parsed = '/media/tim/E/datasets/task_D_D_parsed_combined/validation'


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

max_seq_length = 16
batch_size = 16
num_workers = 10

d_model = 2055 
n_heads = 5

dropout = 0.1
# dim_feedforward
d_ff = 512
n_layers = 3
epochs = 1000
