import torch
import torch.nn as nn
import math
import numpy as np
from warnings import warn
import pdb
from mnets.mnet_interface import MainNetInterface
from utils import init_utils as iutils
from utils.module_wrappers import CLHyperNetInterface
from LSTMs.lstm_hypermodel import RNN


class ChunkedHyperNetworkHandler(nn.Module, CLHyperNetInterface):

    def __init__(self, target_shapes, num_tasks, chunk_dim=2586,
                 layers=[50, 100], te_dim=8, activation_fn=torch.nn.ReLU(),
                 use_bias=True, no_weights=False, ce_dim=None,
                 init_weights=None, dropout_rate=-1, noise_dim=-1,
                 temb_std=-1, no_te_embs=False):
        # FIXME find a way using super to handle multiple inheritence.
        # super(ChunkedHyperNetworkHandler, self).__init__()
        nn.Module.__init__(self)
        CLHyperNetInterface.__init__(self)
        assert (len(target_shapes) > 0)
        assert (init_weights is None or no_weights is False)
        assert (ce_dim is not None)
        print('7-------init in chunked_hyperhandler-------------')
        self._target_shapes = target_shapes
        self._num_tasks = num_tasks
        self._no_te_embds = no_te_embs
        self._ce_dim = ce_dim
        self._chunk_dim = chunk_dim
        self._layers = layers
        self._use_bias = use_bias
        self._act_fn = activation_fn
        self._init_weights = init_weights
        self._no_weights = no_weights
        self._te_dim = te_dim
        self._noise_dim = noise_dim
        self._temb_std = temb_std
        self._shifts = None
        #please update the hidden_state size and lstm layers here
        self._lstm_hidden_size = 96
        self._lstm_layers = 1
        # FIXME: weights should incorporate chunk embeddings as they are part of
        # theta.
        if init_weights is not None:
            warn('Argument "init_weights" does not yet allow initialization ' +
                 'of chunk embeddings.')
        self._num_outputs = MainNetInterface.shapes_to_num_weights(self._target_shapes)
        ### Generate embeddings for all weight chunks.
        self._num_chunks = int(np.ceil(self._num_outputs / chunk_dim))
        if no_weights:
            self._embs = None
        else:
            self._embs = nn.Parameter(data=torch.Tensor(self._num_chunks, ce_dim), requires_grad=True)
            nn.init.normal_(self._embs, mean=0., std=1.)
            # Task embeddings.

        print('------------------------------------------------------------')
        self._lstms = RNN(ce_dim,# + te_dim,
                          self._lstm_hidden_size, self._lstm_layers, chunk_dim, self._target_shapes,
                          num_tasks, te_dim=te_dim,
                          use_bias=use_bias, no_weights=no_weights, init_weights=init_weights,
                          ce_dim=ce_dim + (noise_dim if noise_dim != -1 else 0), dropout_rate=dropout_rate,
                          noise_dim=-1,
                          temb_std=temb_std, no_te_embs=False, init_states=None)

        self._num_weights, self._theta_shapes = self._lstms.get_n_params(self._lstms)

        # print('self._num_weights, self._theta_shapes',self._num_weights, self._theta_shapes)
        # print('self._num_weights1, self._theta_shapes1',self._num_weights1, self._theta_shapes1)
        self._theta_shapes = self._theta_shapes[self._num_tasks:]
        self.li = list(self._lstms.parameters())
        self._theta_lstm = self.li[self._num_tasks:]
        # for p in list(self._theta_lstm):
        #   print('---', p.shape)
        # print('self._lstms', self._lstms)
        # print('forward_method',forward_method)
        print('------------------------------------------------------------------')

        # Note, the chunk embeddings are part of theta.
        hdims = self._theta_shapes
        ntheta = MainNetInterface.shapes_to_num_weights(hdims) + (self._embs.numel() if not no_weights else 0)

        ntembs = int(np.sum([t.numel() for t in self.get_task_embs()]))
        self._num_weights = ntheta + ntembs
        print('The hypernetwork has a total of %d outputs.' % self._num_outputs)
        # [67,32][7000,64][7000]
        self._theta_shapes = [[self._num_chunks, ce_dim]] + \
                             self._theta_shapes

        self._is_properly_setup()

    @property
    def chunk_embeddings(self):
        """Getter for read-only attribute :attr:`chunk_embeddings`.

        Get the chunk embeddings used to produce a full set of main network
        weights with the underlying (small) hypernetwork.

        Returns:
            A list of all chunk embedding vectors.
        """
        # print('def chunk_embeddings(self) called in chunked')
        return list(torch.split(self._embs, 1, dim=0))

    # @override from CLHyperNetInterface
    def forward(self, task_id=None, theta=None, dTheta=None, task_emb=None,
                ext_inputs=None, squeeze=True, test=False):
        # print('8------------forward in chunked-----------------------')
        # pdb.set_trace()
        # print('theta',theta.shape)
        # if theta is not None:
        # print('its here')

        '''
        # if we want to freeze weights use this code
        count = 0
        for p in self._lstms.parameters():
            if count > (self._num_tasks - 1):
                if count < (self._num_tasks + 1):
                    ingate, forgetgate, cellgate, outgate = torch.chunk(p, 4)
                    # ingate.detach()
                    #print('ingate', ingate)
                    # ingate.no
                    # ingate.requires_grad = False
                    # print(ingate.requires_grad)
                    # print(ingate.shape)
            count = count + 1
        # for p in list(self._lstms.parameters()):
        #   print(p.shape)

        for p in list(self._lstms[1:5]):
            inputg, forgetg, cellg, outputg = torch.chunk(p, 4)
            print('input', input.)

            print(x.shape)
            print(y.shape)
            print(z.shape)
            print(w.shape)
        '''
        # freezing code ends here

        if task_id is None and task_emb is None:
            raise Exception('The hyper network has to get either a task ID' +
                            'to choose the learned embedding or directly ' +
                            'get an embedding as input (e.g. from a task ' +
                            'recognition model).')

        if not self.has_theta and theta is None:
            raise Exception('Network was generated without internal weights. ' +
                            'Hence, "theta" option may not be None.')

        if ext_inputs is not None:
            # FIXME If this will be implemented, please consider:
            # * batch size will have to be multiplied based on num chunk
            #   embeddings and the number of external inputs -> large batches
            # * noise dim must adhere correct behavior (different noise per
            #   external input).
            raise NotImplementedError('This hypernetwork implementation does ' +
                                      'not yet support the passing of external inputs.')

        if theta is None:
            theta = self.theta

        else:
            assert (len(theta) == len(self.theta_shapes))
            assert (np.all(np.equal(self._embs.shape, list(theta[0].shape))))

        chunk_embs = theta[0]
        hnet_theta = theta[1:]

        if dTheta is not None:
            assert (len(dTheta) == len(self.theta_shapes))
            chunk_embs = chunk_embs + dTheta[0]
            hnet_dTheta = dTheta[1:]
        else:
            hnet_dTheta = None

        # Concatenate the same noise to all chunks, such that it can be
        # viewed as if it were an external input.
        if self._noise_dim != -1:
            if self.training:
                eps = torch.randn((1, self._noise_dim))
            else:
                eps = torch.zeros((1, self._noise_dim))
            if self._embs.is_cuda:
                eps = eps.to(self._embs.get_device())

            eps = eps.expand(self._num_chunks, self._noise_dim)
            chunk_embs = torch.cat([chunk_embs, eps], dim=1)

        weights = self._lstms.forward(task_id=task_id, theta=hnet_theta, dTheta=None, task_emb=task_emb,
                                      ext_inputs=chunk_embs, test=test)
        weights = torch.stack(weights)
        # weights=weights
        weights1 = []
        weights1.append(weights)
        weights = weights1[0].view(1, -1)
        ind = 0
        ret = []
        for j, s in enumerate(self.target_shapes):
            num = int(np.prod(s))
            W = weights[0][ind:ind + num]
            ind += num
            W = W.view(*s)
            if self._shifts is not None:  # FIXME temporary test!
                W += self._shifts[j]
            ret.append(W)
        return ret

    # @override from CLHyperNetInterface
    @property
    def theta(self):

        return [self._embs] + self._theta_lstm

    # @override from CLHyperNetInterface
    def get_task_embs(self):
        """Overriden super class method."""
        return self._lstms.get_task_embs()

    # @override from CLHyperNetInterface
    def get_task_emb(self, task_id):
        """Overriden super class method."""
        return self._lstms.get_task_emb(task_id)

    # @override from CLHyperNetInterface
    @property
    def has_theta(self):
        """Getter for read-only attribute ``has_theta``."""
        return not self._no_weights

    # @override from CLHyperNetInterface
    @property
    def has_task_embs(self):
        """Getter for read-only attribute ``has_task_embs``."""
        return self._lstms.has_task_embs

    # @override from CLHyperNetInterface
    @property
    def num_task_embs(self):
        """Getter for read-only attribute ``num_task_embs``."""
        return self._lstms.num_task_embs

    def apply_chunked_hyperfan_init(self, method='in', use_xavier=False,
                                    temb_var=1., ext_inp_var=1., eps=1e-5,
                                    cemb_normal_init=False):
        r"""Initialize the network using a chunked hyperfan init.

        Inspired by the method
        `Hyperfan Init <https://openreview.net/forum?id=H1lma24tPB>`__ which we
        implemented for the full hypernetwork in method
        :meth:`toy_example.hyper_model.HyperNetwork.apply_hyperfan_init`, we
        heuristically developed a better initialization method for chunked
        hypernetworks.

        Unfortunately, the `Hyperfan Init` method does not apply to this kind of
        hypernetwork, since we reuse the same hypernet output head for the whole
        main network.

        Luckily, we can provide a simple heuristic. Similar to
        `Meyerson & Miikkulainen <https://arxiv.org/abs/1906.00097>`__ we play
        with the variance of the input embeddings to affect the variance of the
        output weights.

        In a chunked hypernetwork, the input for each chunk is identical except
        for the chunk embeddings :math:`\mathbf{c}`. Let :math:`\mathbf{e}`
        denote the remaining inputs to the hypernetwork, which are identical
        for all chunks. Then, assuming the hypernetwork was initialized via
        fan-in init, the variance of the hypernetwork output :math:`\mathbf{v}`
        can be written as follows (see documentation of method
        :meth:`toy_example.hyper_model.HyperNetwork.apply_hyperfan_init`):

        .. math::

            \text{Var}(v) = \frac{n_e}{n_e+n_c} \text{Var}(e) + \
                \frac{n_c}{n_e+n_c} \text{Var}(c)

        Hence, we can achieve a desired output variance :math:`\text{Var}(v)`
        by initializing the chunk embeddinggs :math:`\mathbf{c}` via the
        following variance:

        .. math::

            \text{Var}(c) = \max \Big\{ 0, \
                \frac{1}{n_c} \big[ (n_e+n_c) \text{Var}(v) - \
                n_e \text{Var}(e) \big] \Big\}

        Now, one important question remains. How do we pick a desired output
        variance :math:`\text{Var}(v)` for a chunk?

        Note, a chunk may include weights from several layers. The likelihood
        for this to happen depends on the main net architecture and the chunk
        size (see constructor argument ``chunk_dim``). The smaller the chunk
        size, the less likely it is that a chunk will contain elements from
        multiple main net weight tensors.

        In case each chunk would contain only weights from one main net weight
        tensor, we could simply pick the variance :math:`\text{Var}(v)` that
        would have been chosen by a main net initialization method (such as
        Xavier).

        In case a chunk contains contributions from several main net weight
        tensors, we apply the following heuristic. If a chunk contains
        contributions of a set of main network weight tensors
        :math:`W_1, \dots, W_K` with relative contribution sizes\
        :math:`n_1, \dots, n_K` such that :math:`n_1 + \dots + n_K = n_v` where
        :math:`n_v` denotes the chunk size and if the corresponding main network
        initialization method would require init varainces
        :math:`\text{Var}(w_1), \dots, \text{Var}(w_K)`, then we simply request
        a weighted average as follow:

        .. math::

            \text{Var}(v) = \frac{1}{n_v} \sum_{k=1}^K n_k \text{Var}(w_k)

        What about bias vectors? Usually, the variance analysis applied to
        Xavier or Kaiming init assumes that biases are initialized to zero. This
        is not possible in this setting, as it would require assigning a
        negative variance to :math:`\mathbf{c}`. Instead, we follow the default
        PyTorch initialization (e.g., see method ``reset_parameters`` in class
        :class:`torch.nn.Linear`). There, bias vectors are initialized uniformly
        within a range of :math:`\pm \frac{1}{\sqrt{f_{\text{in}}}}` where
        :math:`f_{\text{in}}` refers to the fan-in of the layer. This type of
        initialization corresponds to a variance of
        :math:`\text{Var}(v) = \frac{1}{3 f_{\text{in}}}`.

        Warning:
            Note, in order to compute the fan-in of layers with bias vectors, we
            need access to the corresponding weight tensor in the same layer.
            Since there is no clean way of matching a bias shape to its
            corresponging weight tensor shape we use the following heuristic,
            which should be correct for most main networks. We assume that the
            shape directly preceding a bias shape in the constructor argument
            ``target_shapes`` is the corresponding weight tensor.

        Note:
            Constructor argument ``noise_dim`` is automatically considered by
            this method.

        Note:
            We hypernet inputs should be zero mean.

        Warning:
            This method considers all 1D target weight tensors as bias vectors.

        Note:
            To avoid that the variances with which chunks are initialized
            have to be clipped (because they are too small or even negative),
            the variance of the remaining hypernet inputs should be properly
            scaled. In general, one should adhere the following rule

            .. math::

                \text{Var}(e) < \frac{n_e+n_c}{n_e} \text{Var}(v)

            This method will calculate and print the maximum value that should
            be chosen for :math:`\text{Var}(e)` and will print warnings if
            variances have to be clipped.

        Args:
            method (str): The type of initialization that should be applied.
                Possible options are:

                - ``in``: Use `Chunked Hyperfan-in`, i.e., rather the output
                  variances of the hypernetwork should correspond to fan-in
                  variances.
                - ``out``: Use `Chunked Hyperfan-out`, i.e., rather the output
                  variances of the hypernetwork should correspond to fan-out
                  variances.
                - ``harmonic``: Use the harmonic mean of the fan-in and fan-out
                  variance as target variance of the hypernetwork output.
            use_xavier (bool): Whether Kaiming (``False``) or Xavier (``True``)
                init should be used.
            temb_var (float): The initial variance of the task embeddings.

                .. note::
                    If ``temb_std`` was set in the constructor, then this method
                    will automatically correct the provided ``temb_var`` as 
                    follows: :code:`temb_var += temb_std**2`.
            ext_inp_var (float): The initial variance of the external input.
                Only needs to be specified if external inputs are provided.
                
                .. note::
                    Not supported yet by this hypernetwork type, but should soon
                    be included as a feature.
            eps (float): The minimum variance with which a chunk embedding is
                initialized.
            cemb_normal_init (bool): Use normal init for chunk embeedings
                rather than uniform init.
        """
        print('hyperfan init in chunked hyper')
        if method not in ['in', 'out', 'harmonic']:
            raise ValueError('Invalid value for argument "method".')
        if not self.has_theta:
            raise ValueError('Hypernet without internal weights can\'t be ' +
                             'initialized.')

        ### Compute input variance ###
        # The input variance does not include the variance of chunk embeddings!
        # Instead, it is the varaince of the inputs that are shared across all
        # chunks.
        if self._temb_std != -1:
            # Sum of uncorrelated variables.
            temb_var += self._temb_std ** 2

        assert self._noise_dim == -1 or self._noise_dim > 0

        # TODO external inputs are not yet considered.
        inp_dim = self._te_dim + \
                  (self._noise_dim if self._noise_dim != -1 else 0)
        # (self._size_ext_input if self._size_ext_input is not None else 0) \

        inp_var = (self._te_dim / inp_dim) * temb_var
        # if self._size_ext_input is not None:
        #    inp_var += (self._size_ext_input  / inp_dim) * ext_inp_var
        if self._noise_dim != -1:
            inp_var += (self._noise_dim / inp_dim) * 1.

        c_dim = self._ce_dim

        ### Compute target variance of each output tensor ###
        target_vars = []

        for i, s in enumerate(self.target_shapes):
            # FIXME 1D shape is not necessarily bias vector.
            if len(s) == 1:  # Assume it's a bias vector
                # Assume that last shape has been the corresponding weight
                # tensor.
                if i > 0 and len(self.target_shapes[i - 1]) > 1:
                    fan_in, _ = iutils.calc_fan_in_and_out( \
                        self.target_shapes[i - 1])
                else:
                    # FIXME Quick-fix, use fan-out instead.
                    fan_in = s[0]

                target_vars.append(1. / (3. * fan_in))

            else:
                fan_in, fan_out = iutils.calc_fan_in_and_out(s)

                c_relu = 1 if use_xavier else 2

                var_in = c_relu / fan_in
                var_out = c_relu / fan_out

                if method == 'in':
                    var = var_in
                elif method == 'out':
                    var = var_out
                else:
                    var = 2 * (1. / var_in + 1. / var_out)

                target_vars.append(var)

        ### Target variance per chunk ###
        chunk_vars = []
        i = 0
        n = np.prod(self.target_shapes[i])

        for j in range(self._num_chunks):
            m = self._chunk_dim
            var = 0

            while m > 0:
                # Special treatment to fill up last chunk.
                if j == self._num_chunks - 1 and i == len(target_vars) - 1:
                    assert n <= m
                    o = m
                else:
                    o = min(m, n)

                var += o / self._chunk_dim * target_vars[i]
                m -= o
                n -= o

                if n == 0:
                    i += 1
                    if i < len(target_vars):
                        n = np.prod(self.target_shapes[i])

            chunk_vars.append(var)

        max_inp_var = (inp_dim + c_dim) / inp_dim * min(chunk_vars)
        max_inp_std = math.sqrt(max_inp_var)
        print('Initializing hypernet with Chunked Hyperfan Init ...')
        if inp_var >= max_inp_var:
            warn('Note, hypernetwork inputs should have an initial total ' +
                 'variance (std) smaller than %f (%f) in order for this ' \
                 % (max_inp_var, max_inp_std) + 'method to work properly.')

        ### Compute variances of chunk embeddings ###
        # We could have done that in the previous loop. But I think the code is
        # more readible this way.
        c_vars = []
        n_clipped = 0
        for i, var in enumerate(chunk_vars):
            c_var = 1. / c_dim * ((inp_dim + c_dim) * var - inp_dim * inp_var)
            if c_var < eps:
                n_clipped += 1
                # warn('Initial variance of chunk embedding %d has to ' % i + \
                #     'be clipped.')

            c_vars.append(max(eps, c_var))

        if n_clipped > 0:
            warn('Initial variance of %d/%d ' % (n_clipped, len(chunk_vars)) + \
                 'chunk embeddings had to be clipped.')

        ### Initialize chunk embeddings ###
        for i, c_emb in enumerate(self.chunk_embeddings):
            c_std = math.sqrt(c_vars[i])
            if cemb_normal_init:
                torch.nn.init.normal_(c_emb, mean=0, std=c_std)
            else:
                a = math.sqrt(3.0) * c_std
                torch.nn.init._no_grad_uniform_(c_emb, -a, a)

        ### Initialize hypernet with fan-in init ###
        for i, w in enumerate(self._lstm.theta):
            if w.ndim == 1:  # bias
                assert i % 2 == 1
                torch.nn.init.constant_(w, 0)

            else:
                if use_xavier:
                    iutils.xavier_fan_in_(w)
                else:
                    torch.nn.init.kaiming_uniform_(w, mode='fan_in',
                                                   nonlinearity='relu')


if __name__ == '__main__':
    pass
