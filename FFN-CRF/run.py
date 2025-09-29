from FFN_CRF import FFNCRF
from DataLoader import DataLoader

train_data = DataLoader(r'Data\little-data\train.conll')
eval_data = DataLoader(r'Data\little-data\dev.conll')

model = FFNCRF([30,30], init_data=train_data)

print(model.model.sizes)
model.SGD(12,30,2,evaluation_data=eval_data)