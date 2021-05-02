import time

import models
import torch
import torch.optim as optim
import util
from torch.autograd import Variable

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class Evaluator():
    def __init__(self, data_loader, logger, config):
        self.loss_meters = util.AverageMeter()
        self.acc_meters = util.AverageMeter()
        self.acc5_meters = util.AverageMeter()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.data_loader = data_loader
        self.logger = logger
        self.log_frequency = config.log_frequency if config.log_frequency is not None else 100
        self.config = config
        self.current_acc = 0
        self.current_acc_top5 = 0
        self.confusion_matrix = torch.zeros(config.num_classes, config.num_classes)
        return

    def _reset_stats(self):
        self.loss_meters = util.AverageMeter()
        self.acc_meters = util.AverageMeter()
        self.acc5_meters = util.AverageMeter()
        self.confusion_matrix = torch.zeros(self.config.num_classes, self.config.num_classes)
        return

    def eval(self, epoch, model):
        model.eval()
        for i, (images, labels) in enumerate(self.data_loader["test_dataset"]):
            start = time.time()
            log_payload = self.eval_batch(images=images, labels=labels, model=model)
            end = time.time()
            time_used = end - start
        display = util.log_display(epoch=epoch,
                                   global_step=i,
                                   time_elapse=time_used,
                                   **log_payload)
        if self.logger is not None:
            self.logger.info(display)
        return

    def eval_batch(self, images, labels, model):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with torch.no_grad():
            pred = model(images)
            loss = self.criterion(pred, labels)
            acc, acc5 = util.accuracy(pred, labels, topk=(1, 5))
            _, preds = torch.max(pred, 1)
            for t, p in zip(labels.view(-1), preds.view(-1)):
                self.confusion_matrix[t.long(), p.long()] += 1

        self.loss_meters.update(loss.item(), n=images.size(0))
        self.acc_meters.update(acc.item(), n=images.size(0))
        self.acc5_meters.update(acc5.item(), n=images.size(0))
        payload = {"acc": acc.item(),
                   "acc_avg": self.acc_meters.avg,
                   "acc5": acc5.item(),
                   "acc5_avg": self.acc5_meters.avg,
                   "loss": loss.item(),
                   "loss_avg": self.loss_meters.avg}
        return payload

    def _pgd_whitebox(self, model, X, y, random_start=True, epsilon=0.031, num_steps=20, step_size=0.003):
        model.eval()
        out = model(X)
        acc = (out.data.max(1)[1] == y.data).float().sum()
        X_pgd = Variable(X.data, requires_grad=True)
        if random_start:
            random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
            X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

        for _ in range(num_steps):
            opt = optim.SGD([X_pgd], lr=1e-3)
            opt.zero_grad()

            with torch.enable_grad():
                loss = torch.nn.CrossEntropyLoss()(model(X_pgd), y)
            loss.backward()
            eta = step_size * X_pgd.grad.data.sign()
            X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
            eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
            X_pgd = Variable(X.data + eta, requires_grad=True)
            X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
        acc_pgd = (model(X_pgd).data.max(1)[1] == y.data).float().sum()
        return acc.item(), acc_pgd.item()
