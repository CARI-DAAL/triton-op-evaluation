print('before_torch',flush=True)
import torch
print('before_torch_npu',flush=True)
import torch_npu
print('before_tensor',flush=True)
x=torch.ones(4,device='npu')
torch.npu.synchronize()
print('tensor_ready',str(x.device),float(x.sum().cpu()),flush=True)
