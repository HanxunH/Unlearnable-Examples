import torch
import torch.nn as nn
import torch.nn.functional as F
import models
from torch.autograd import Variable
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class MadrysLoss(nn.Module):
    def __init__(self, step_size=0.007, epsilon=0.031, perturb_steps=10, distance='l_inf', cutmix=False):
        super(MadrysLoss, self).__init__()
        self.step_size = step_size
        self.epsilon = epsilon
        self.perturb_steps = perturb_steps
        self.distance = distance
        self.cross_entropy = models.CutMixCrossEntropyLoss() if cutmix else torch.nn.CrossEntropyLoss()

    def forward(self, model, x_natural, y, optimizer):
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

        # generate adversarial example
        x_adv = x_natural.clone() + self.step_size * torch.randn(x_natural.shape).to(device)
        if self.distance == 'l_inf':
            for _ in range(self.perturb_steps):
                x_adv.requires_grad_()
                loss_ce = self.cross_entropy(model(x_adv), y)
                grad = torch.autograd.grad(loss_ce, [x_adv])[0]
                x_adv = x_adv.detach() + self.step_size * torch.sign(grad.detach())
                x_adv = torch.min(torch.max(x_adv, x_natural - self.epsilon), x_natural + self.epsilon)
                x_adv = torch.clamp(x_adv, 0.0, 1.0)
        else:
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

        for param in model.parameters():
            param.requires_grad = True

        model.train()
        # x_adv = Variable(x_adv, requires_grad=False)
        optimizer.zero_grad()
        logits = model(x_adv)
        loss = self.cross_entropy(logits, y)

        return logits, loss
