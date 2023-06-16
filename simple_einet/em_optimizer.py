import torch


class EmOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=0.1, grad_acc_step=1):

        self.lr = lr
        self.grad_acc_step = grad_acc_step
        self.curr_grad_acc_step = 0
        defaults = dict(lr=lr)
        super().__init__(params, defaults)

    def step(self, closure=None):
        self.curr_grad_acc_step += 1
        if self.curr_grad_acc_step % self.grad_acc_step == 0:
            for group in self.param_groups:
                if group["is_leaf"]:
                    for param in group["params"]:
                        p = param.leaf.ll.grad
                        weighted_stats = (
                            p.unsqueeze(-1) * param.leaf.suff_stats
                        ).sum(0)
                        p = p.sum(0)
                        param.leaf.ll.grad.zero_()
                        param.leaf.params.data = (
                            1.0 - self.lr
                        ) * param.leaf.params + self.lr * (weighted_stats / (p.unsqueeze(-1) + 1e-12))
                        param.leaf.project_params()

                else:
                    for param in group["params"]:
                        if param.dim() == 6:  # einsum weights
                            normalization_dims = [4, 5]
                        elif param.dim() == 5:  # mixing weights
                            normalization_dims = [4]
                        else:                   # mixing layer
                            normalization_dims = [2]
                            return
                        n = param.grad * param.data
                        p = torch.clamp(n, 1e-16)
                        p = p / (p.sum(normalization_dims, keepdim=True))
                        param.data = (1.0 - self.lr) * param + self.lr * p

                        param.data = torch.clamp(param, 1e-16)
                        param.data = param / \
                            (param.sum(normalization_dims, keepdim=True))
                        param.grad = None
