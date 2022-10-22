import torch
import torch.nn as nn
import torchvision
import numpy as np
import math
import torchvision.transforms as transforms
from LSTMs.Lstm_interface import LstmHyperNetInterface
from mnets.mnet_interface import MainNetInterface
from LSTMs.LSTM_cell import LSTMCell

"""
This cell will intialise the LSTM Cell and operations can be found in LSTM_cell.py

"""
class RNN(nn.Module, LstmHyperNetInterface):
    def __init__(self, input_size, hidden_size, num_layers,
                 chunk_size, target_shapes, num_tasks,
                 te_dim=32,
                 use_bias=True, no_weights=False, ce_dim=None,
                 init_weights=None, dropout_rate=-1, noise_dim=-1,
                 temb_std=-1, no_te_embs=False, init_states=None):
        # super(RNN, self).__init__()
        nn.Module.__init__(self)
        LstmHyperNetInterface.__init__(self)
        self.num_layers = num_layers
        self._input_size = input_size
        self._hidden_size = hidden_size
        self._theta = None
        self._init_states = None
        self._num_weights = None
        self._theta_shapes = None
        self._num_outputs = None
        self._chunk_size = chunk_size
        self._size_ext_input = None
        assert (len(target_shapes) > 0)
        assert (init_weights is None or no_weights is False)
        assert (ce_dim is not None)
        self._target_shapes = target_shapes
        self._num_tasks = num_tasks
        self._ce_dim = ce_dim
        self._use_bias = use_bias
        self._init_weights = init_weights
        self._no_weights = no_weights
        self._te_dim = te_dim
        self._noise_dim = noise_dim
        self._temb_std = temb_std
        self._shifts = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._num_outputs = MainNetInterface.shapes_to_num_weights(self._target_shapes)
        self._num_chunks = int(np.ceil(self._num_outputs / self._chunk_size))
        print('self._num_chunks ', self._num_chunks)
        if no_te_embs:
            self._task_embs = None
        else:
            self._task_embs = nn.ParameterList()
            for _ in range(0, self._num_tasks):
                self._task_embs.append(nn.Parameter(data=torch.Tensor(te_dim), requires_grad=True))
                torch.nn.init.normal_(self._task_embs[-1], mean=0., std=1.)
        self.lstm = LSTMCell(input_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, chunk_size)
        self._num_outputs = MainNetInterface.shapes_to_num_weights(self.target_shapes)

    def forward(self, task_id=None, theta=None, dTheta=None, task_emb=None, ext_inputs=None, squeeze=True, test=False):

        if task_id is None and task_emb is None:
            raise Exception('The hyper network has to get either a task ID' +
                            'to choose the learned embedding or directly ' +
                            'get an embedding as input (e.g. from a task ' +
                            'recognition model).')

        # Select task embeddings.
        if not self.has_task_embs and task_emb is None:
            raise Exception('The network was created with no internal task ' +
                            'embeddings, thus parameter "task_emb" has to ' +
                            'be specified.')

        if task_emb is None:
            task_emb = self._task_embs[task_id]
        if self.training and self._temb_std != -1:
            task_emb.add(torch.randn_like(task_emb) * self._temb_std)
        if self._noise_dim != -1:
            if self.training:
                eps = torch.randn((1, self._noise_dim))
            else:
                eps = torch.zeros((1, self._noise_dim))
            if self._embs.is_cuda:
                eps = eps.to(self._embs.get_device())

            eps = eps.expand(self._num_chunks, self._noise_dim)
            ext_inputs = torch.cat([ext_inputs, eps], dim=1)

        if ext_inputs is not None:
            batch_size = ext_inputs.shape[0]
            task_emb = task_emb.expand(batch_size, self._te_dim)
            h = torch.cat([task_emb, ext_inputs], dim=1)
            #h = ext_inputs
        if self._noise_dim != -1:
            if self.training:
                eps = torch.randn((batch_size, self._noise_dim))
            else:
                eps = torch.zeros((batch_size, self._noise_dim))
            if h.is_cuda:
                eps = eps.to(h.get_device())
            h = torch.cat([h, eps], dim=1)
        main_net_weights = []
        sequence_length = 1
        init_states = None

        for i in range(0, self._num_chunks):
            inp = h[i].reshape(-1, sequence_length, self._input_size).to(self.device)
            # print('inp',inp)
            if init_states is None:
                h_t = torch.zeros(self.num_layers, inp.size(0), self._hidden_size).to(self.device)
                c_t = torch.zeros(self.num_layers, inp.size(0), self._hidden_size).to(self.device)
                # print('h_t, c_t', h_t, c_t)
                c_t = c_t[0, :, :]
                h_t = h_t[0, :, :]
                init_states = 1
            out = []
            for seq in range(inp.size(1)):
                # print('seq',seq)
                h_t, c_t = self.lstm(inp[:, seq, :], (h_t, c_t), task_id, theta, test=test)

            #if task_id > 0:
            #self.fc.weight.requires_grad = False
            #self.fc.bias.requires_grad = False

            if not test:
                out = self.fc(h_t)
            if test:
                out = torch.matmul(h_t, (torch.transpose(theta[16], 0, 1))) + theta[17]
            # out = self.fc(out)
            we = np.squeeze(out)
            # we.clone().detach().requires_grad_(True)
            # we=torch.tensor(we,requires_grad=True)
            # we.retain_grad()
            main_net_weights.append(we)
        return main_net_weights

    def get_n_params(self, model1):
        pp = 0
        param = []
        for p in list(model1.parameters()):
            nn = 1
            # param.append(p.shape)
            param.append(list(p.shape))
            for s in list(p.size()):
                nn = nn * s
            pp += nn
        return pp, param
