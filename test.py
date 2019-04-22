import pickle as pkl
import numpy as np

with open('/Users/william/Google Drive/STUDY/Columbia 2019 Spring/RL8100/Project/Finance/Portfolio_RL/Data/input_tensor.pkl','rb') as f:
        #4 dimensional tensor
        #[batch,]
        data = pkl.load(f)

state = np.transpose(data,(1,2,0))[None,...]
print(state.shape)
vt = state[0,:,-1,3]
vt_1 = state[0,:,-2,3]

print(vt)

print(vt_1)
print(data[0].shape)
