import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

'''
    Paths to downlaoded datasets:
    CALVIN dataset can be found here: https://github.com/mees/calvin/tree/main/dataset
    datasets can be parsed by executing 'parse_dataset.py' 
'''

datapath_training = '/media/tim/E/datasets/task_D_D/training'
datapath_val = '/media/tim/E/datasets/task_D_D/validation'

datapath_training_parsed = '/media/tim/E/datasets/task_D_D_parsed/training'
datapath_val_parsed = '/media/tim/E/datasets/task_D_D_parsed/validation'

datapath_training_abcd = '/media/tim/E/task_ABC_D/training'
datapath_val_abcd  = '/media/tim/E/task_ABC_D/validation'

datapath_training_abcd_parsed = '/media/tim/E/datasets/task_ABC_D_parsed/training'
datapath_val_abcd_parsed = '/media/tim/E/datasets/task_ABC_D_parsed/validation'


'''
    Captioning Model Parameters
'''
max_seq_length = 16
batch_size = 12
num_workers = 10
d_model = 2055 
#d_model = 2070
n_heads = 5
#n_heads = 6
dropout = 0.1
# dim_feedforward
d_ff = 512
n_layers = 3
epochs = 1000
start_epi_dd = 0
end_epi_dd = 611098
img_size = 180

