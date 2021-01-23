import torch


def average(outputs):
    return sum(outputs) / len(outputs)


def summation(outputs, factor):
    return sum([output * factor for output in outputs])


def onehot_encoding(label, max_val):
    label = label.view(-1)
    onehot = torch.zeros(label.size(0), max_val).float().to(label.device)
    onehot.scatter_(1, label.view(-1, 1), 1)
    return onehot
