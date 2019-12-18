import torch
import torch.nn as nn
from collections import OrderedDict


class SDT(nn.Module):
    
    def __init__(self, args):
        super(SDT, self).__init__()
        self.args = args
        self.device = torch.device('cuda' if self.args['cuda'] else 'cpu')
        self.inner_node_num = 2 ** self.args['depth'] - 1
        self.leaf_node_num = 2 ** self.args['depth']
        self.penalty_list = [args['lamda'] * (2 ** (-depth)) for depth in range(0, self.args['depth'])] 
        self.inner_nodes = nn.Sequential(OrderedDict([
                        ('linear', nn.Linear(self.args['input_dim']+1, self.inner_node_num, bias=False)),
                        ('sigmoid', nn.Sigmoid()),
                        ]))
        self.leaf_nodes = nn.Linear(self.leaf_node_num, self.args['output_dim'], bias=False)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.args['lr'], weight_decay=self.args['weight_decay'])
    
    def forward(self, data):
        mu, penalty = self._forward(data)
        output = self.leaf_nodes(mu)
        return output, penalty
    
    """ Core implementation on data forwarding in SDT """
    def _forward(self, data):
        batch_size = data.size()[0]
        data = self._data_augment_(data)
        path_prob = self.inner_nodes(data)
        path_prob = torch.unsqueeze(path_prob, dim=2)
        path_prob = torch.cat((path_prob, 1-path_prob), dim=2)
        _mu = data.data.new(batch_size,1,1).fill_(1.)
        _penalty = torch.tensor(0.).to(self.device)
        
        begin_idx = 0
        end_idx = 1
        
        for layer_idx in range(0, self.args['depth']):
            _path_prob = path_prob[:, begin_idx:end_idx, :]
            _penalty = _penalty + self._cal_penalty(layer_idx, _mu, _path_prob)
            _mu = _mu.view(batch_size, -1, 1).repeat(1, 1, 2)
            _mu = _mu * _path_prob
            begin_idx = end_idx
            end_idx = begin_idx + 2 ** (layer_idx+1)
        mu = _mu.view(batch_size, self.leaf_node_num)
        return mu, _penalty          
    
    """ Calculate penalty term for inner-nodes in different layer """
    def _cal_penalty(self, layer_idx, _mu, _path_prob):
        penalty = torch.tensor(0.).to(self.device)     
        batch_size = _mu.size()[0]
        _mu = _mu.view(batch_size, 2**layer_idx)
        _path_prob = _path_prob.view(batch_size, 2**(layer_idx+1))
        for node in range(0, 2**(layer_idx+1)):
            alpha = torch.sum(_path_prob[:, node]*_mu[:,node//2], dim=0) / torch.sum(_mu[:,node//2], dim=0)
            penalty = penalty - self.penalty_list[layer_idx] * 0.5 * (torch.log(alpha) + torch.log(1-alpha))
        return penalty
    
    """ Add constant 1 onto the front of each instance """
    def _data_augment_(self, input):
        batch_size = input.size()[0]
        input = input.view(batch_size, -1)
        bias = torch.ones(batch_size, 1).to(self.device)
        input = torch.cat((bias, input), 1)
        return input
    
    def batch_train(self, input, target):
        self.optimizer.zero_grad()
        output, penalty = self.forward(input)
        loss = self.criterion(output, target)
        loss = loss + penalty
        loss.backward()
        self.optimizer.step()
