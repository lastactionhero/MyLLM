# %%
import torch
torch.cuda.is_available()
# %%
tensor0d = torch.tensor(1)
# %%
tensor1d = torch.tensor([1, 2, 3])
# %%
tensor2d = torch.tensor([[1, 2, 3], [4, 5, 6]])
# %%
tensor3d = torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
# %%
tensor4d = torch.tensor([[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], [[[13, 14, 15], [16, 17, 18]], [[19, 20, 21], [22, 23, 24]]]])
# %%
import torch.nn.functional as F
from torch.autograd import grad

y = torch.tensor([1.0])
x1 = torch.tensor([1.1])
w1 = torch.tensor([2.2], requires_grad=True)
b = torch.tensor([0.0], requires_grad=True)

z = x1 * w1 + b 
a = torch.sigmoid(z)

loss = F.binary_cross_entropy(a, y)

grad_L_w1 = grad(loss, w1, retain_graph=True)   #1
grad_L_b = grad(loss, b, retain_graph=True)
# %%
print(grad_L_w1)
print(grad_L_b)
# %%
