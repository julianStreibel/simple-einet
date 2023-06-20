import torch
import torch.nn.functional as F

# 10 states
w = torch.rand(5, 10)
w_entropy = w.log().sum(dim=1)

wanted_entropy = 5
t = -((w*w.log()).sum(dim=1) / wanted_entropy).view(5, 1)

print("w_entropy", w_entropy)
print("t", t)

softmax = F.softmax(w/t, 1)
softmax_entropy = (softmax*softmax.log()).sum(dim=1)

print("softmax_entropy", softmax_entropy)
