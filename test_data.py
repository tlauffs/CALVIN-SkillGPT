import h5py

hf = h5py.File('/media/tim/D/datasets/lang_table/lang_table_test.h5', 'r')

# print(list(hf.keys()))

data_clip_instruction_features = hf.get('1').get('clip_instruction_features')
data_instruction = hf.get('1').get('instruction')
data_actions = hf.get('1').get('actions')
data_observations = hf.get('1').get('observations')

print('Clip Instruction Features: \n', data_clip_instruction_features[()], '\n')
print('Instruction: \n', data_instruction[()], '\n')
print('Action: \n', data_actions[()], '\n')
print('Observation: \n', data_observations[()], '\n')