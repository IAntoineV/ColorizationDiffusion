import torch

class ConcatModel(torch.nn.Module):
    def __init__(self, module):
        super(ConcatModel, self).__init__()
        self.module = module

    def forward(self, input1, input2, *args):
        input_concat = torch.cat((input1, input2), dim=1)
        return self.module(input_concat, *args)


class ConcatModelNoT(torch.nn.Module):
    def __init__(self, module):
        super(ConcatModelNoT, self).__init__()
        self.module = module

    def forward(self, input1, input2, *args):
        input_concat = torch.cat((input1, input2), dim=1)
        return self.module(input_concat)