import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from warnings import warn

from mnets.mnet_interface import MainNetInterface as MnetAPIV2
from utils.module_wrappers import MainNetInterface
from utils.torch_utils import init_params

class MainNetwork(nn.Module, MainNetInterface):
    """This is a simple fully-connected network, that receives a simple input x
    and outputs a real value y. The goal is to learn a mapping y = f(x) + eps,
    where eps is (white) noise.

    The output mapping does not include a non-linearity, as we wanna map to the
    whole real line.

    .. deprecated:: 1.0
        Please use class :class:`mnets.mlp.MLP` instead.

    Attributes (additional to base class):
    """
    def __init__(self, weight_shapes, activation_fn=torch.nn.ReLU(),
                 use_bias=True, no_weights=False, init_weights=None,
                 dropout_rate=-1, out_fn=None, verbose=True, 
                 use_spectral_norm=False, use_batch_norm=False):
        """Initialize the network.

        Args:
            weight_shapes: A list of list of integers, denoting the shape of
                each parameter tensor in this network. Note, this parameter only
                has an effect on the construction of this network, if
                "no_weights" is False. Otherwise, it is just used to check the
                shapes of the input to the network in the forward method.
            activation_fn: The nonlinearity used in hidden layers. If None, no
                nonlinearity will be applied.
            use_bias: Whether layers may have bias terms.
            no_weights: If set to True, no trainable parameters will be
                constructed, i.e., weights are assumed to be produced ad-hoc
                by a hypernetwork and passed to the forward function.
            init_weights (optional): This option is for convinience reasons.
                The option expects a list of parameter values that are used to
                initialize the network weights. As such, it provides a
                convinient way of initializing a network with a weight draw
                produced by the hypernetwork.
            dropout_rate: If -1, no dropout will be applied. Otherwise a number
                between 0 and 1 is expected, denoting the dropout rate of hidden
                layers.
            out_fn (optional): If provided, this function will be applied to the
                output neurons of the network. Note, this changes the output
                of the forward method.
            verbose: Whether to print the number of weights in the network.
            use_spectral_norm: Use spectral normalization for training.
            use_batch_norm: Whether batch normalization should be used.
        """
        # FIXME find a way using super to handle multiple inheritence.
        #super(MainNetwork, self).__init__()
        nn.Module.__init__(self)
        MainNetInterface.__init__(self)

        warn('Please use class "mnets.mlp.MLP" instead.', DeprecationWarning)

        if use_spectral_norm:
            raise NotImplementedError('Spectral normalization not yet ' +
                                      'implemented for this network.')
        if use_batch_norm:
            raise NotImplementedError('Batch normalization not yet ' +
                                      'implemented for this network.')

        assert(len(weight_shapes) > 0)
        self._all_shapes = weight_shapes
        self._has_bias = use_bias
        self._a_fun = activation_fn
        assert(init_weights is None or no_weights is False)
        self._no_weights = no_weights
        self._dropout_rate = dropout_rate
        self._out_fn = out_fn

        self._has_fc_out = True

        if use_spectral_norm and no_weights:
            raise ValueError('Cannot use spectral norm in a network without ' +
                             'parameters.')

        if use_spectral_norm:
            self._spec_norm = nn.utils.spectral_norm
        else:
            self._spec_norm = lambda x : x # identity


        if verbose:
            print('Creating an MLP with %d weights' \
                      % (MnetAPIV2.shapes_to_num_weights(self._all_shapes))
                      + (', that uses dropout.' if dropout_rate != -1 else '.'))

        if dropout_rate != -1:
            assert(dropout_rate >= 0. and dropout_rate <= 1.)
            self._dropout = nn.Dropout(p=dropout_rate)

        self._weights = None
        if no_weights:
            self._hyper_shapes = self._all_shapes
            self._is_properly_setup()
            return

        ### Define and initialize network weights.
        # Each odd entry of this list will contain a weight Tensor and each
        # even entry a bias vector.
        self._weights = nn.ParameterList()

        for i, dims in enumerate(self._all_shapes):
            self._weights.append(nn.Parameter(torch.Tensor(*dims),
                                              requires_grad=True))

        if init_weights is not None:
            assert(len(init_weights) == len(self._all_shapes))
            for i in range(len(init_weights)):
                assert(np.all(np.equal(list(init_weights[i].shape),
                                       list(self._weights[i].shape))))
                self._weights[i].data = init_weights[i]
        else:
            for i in range(0, len(self._weights), 2 if use_bias else 1):
                if use_bias:
                    init_params(self._weights[i], self._weights[i + 1])
                else:
                    init_params(self._weights[i])

        self._is_properly_setup()

    def forward(self, x, weights=None):
        """Predict the output y given the input x, that is propagated through a
        fully-connected network (using the given weights).

        Args:
            x: The input to the network.
            weights: A list of torch parameter tensors has to be provided, where
                each tensor has the shape as specified by the list
                "weight_shapes" provided to the constructor.
                If "no_weights" was set in the constructor, then this parameter
                is mandatory.
                Note, when provided, internal parameters are not used.

        Returns:
            (tuple): Tuple containing:
    
            - **y**: The output of the network.
            - **h_y** (optional): If `out_fn` was specified in the constructor,
              then this value will be returned. It is the last hidden activation
              (before the `out_fn` has been applied).
        """
        if self._no_weights and weights is None:
            raise Exception('Network was generated without weights. ' +
                            'Hence, "weights" option may not be None.')

        if weights is None:
            weights = self.weights
        else:
            shapes = self.param_shapes
            assert(len(weights) == len(shapes))
            for i, s in enumerate(shapes):
                assert(np.all(np.equal(s, list(weights[i].shape))))

        hidden = x

        if self.has_bias:
            num_layers = len(weights) // 2
            step_size = 2
        else:
            num_layers = len(weights)
            step_size = 1

        for l in range(0, len(weights), step_size):
            W = weights[l]
            if self.has_bias:
                b = weights[l+1]
            else:
                b = None

            hidden = self._spec_norm(F.linear(hidden, W, bias=b))

            # Only for hidden layers.
            if l / step_size + 1 < num_layers:
                if self._dropout_rate != -1:
                    hidden = self._dropout(hidden)
                if self._a_fun is not None:
                    hidden = self._a_fun(hidden)

        if self._out_fn is not None:
            return self._out_fn(hidden), hidden

        return hidden

    @staticmethod
    def mse(predictions, targets):
        """Compute the mean squared error.

        Args:
            predictions: Outputs y from the network.
            targets: Targets corresponding to the predictions.

        Returns:
            MSE between predictions and targets.
        """
        warn('Use torch.nn.functional.mse_loss instead.', DeprecationWarning)

        return F.mse_loss(predictions, targets)

    @staticmethod
    def weight_shapes(n_in=1, n_out=1, hidden_layers=[10, 10], use_bias=True):
        """Compute the tensor shapes of all parameters in a fully-connected
        network.

        Args:
            n_in: Number of inputs.
            n_out: Number of output units.
            hidden_layers: A list of ints, each number denoting the size of a
                hidden layer.
            use_bias: Whether the FC layers should have biases.

        Returns:
            A list of list of integers, denoting the shapes of the individual
            parameter tensors.
        """
        shapes = []

        prev_dim = n_in
        layer_out_sizes = hidden_layers + [n_out]
        for i, size in enumerate(layer_out_sizes):
            shapes.append([size, prev_dim])
            if use_bias:
                shapes.append([size])
            prev_dim = size

        return shapes

    @staticmethod
    def get_reg_masks(weight_shapes, allowed_outputs, device,
                      use_bias=True):
        """Get a binary mask for a weight preserving regularizer (e.g., via a
        quadratic penalty) for a network with a multi-head output.
        When regularizing the weights for task i, all heads j != i don't
        influence that task performance. Hence, the weights connecting to output
        heads j are arbitrary (with respect to the current task) and don't
        have to be regularized.

        Note, this function only applies to networks that have a fully-connected
        layer as output layer.

        Note, it would only be sensible to include such a mask in the
        regularizer. For instance, in EWC the diagonal Fisher estimate is
        naturally zero for unused weights.

        Args:
            weight_shapes: The shapes of all parameter tensors in the network to
                be regularized (e.g., as returned by the method
                "weight_shapes").
                Note, the last two entries in this list are expected to be the
                weights of a final fully-connected layer (a weight matrix and
                a bias vector). Only these tensors can be masked out by this
                function.
                If not "use_bias", then only the last parameter tensor is
                considered (the weight matrix of the final fully-connected
                layer).
            allowed_outputs: A list of indices, denoting the output neurons
                (in the final layer) that belong to the current task. Only the
                weight connecting to these neurons may be regularized. All other
                output weights will be zeroed-out by the returned mask.
            device: PyTorch device to move the masks to.
            use_bias: Whether the final FC layer uses a bias vector (see
                argument "weight_shapes" for more details).

        Returns:
            A list of binary tensors, where each tensor has a shape
            corresponding to "weight_shapes". All values are 1 except for the
            masks corresponding to the output weights, which are only 1 for the
            weights connecting to the output weights.
        """
        reg_mask = [torch.ones(s).to(device) for s in weight_shapes]

        reg_mask[-1] = torch.zeros(weight_shapes[-1]).to(device)
        if use_bias:
            reg_mask[-2] = torch.zeros(weight_shapes[-2]).to(device)

        if use_bias:
            bias_mask = reg_mask[-1] # Output bias
            weight_mask = reg_mask[-2] # Output weight

            bias_mask[allowed_outputs] = 1
        else:
            weight_mask = reg_mask[-1]
        weight_mask[allowed_outputs, :] = 1

        return reg_mask

if __name__ == '__main__':
    pass
