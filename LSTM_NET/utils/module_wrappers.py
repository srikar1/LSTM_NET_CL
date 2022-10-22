from abc import ABC, abstractmethod
import numpy as np
from warnings import warn

class CLHyperNetInterface(ABC):

    def __init__(self):
        """Initialize the network."""
        super(CLHyperNetInterface, self).__init__()

        warn('Please use class "hnets.hnet_interface.CLHyperNetInterface" ' +
             'instead.', DeprecationWarning)
        #print('init in modulewrappers for class CLHyperNetInterface')
        # The following member variables have to be set by all classes that
        # implement this interface.
        self._theta = None
        self._task_embs = None
        self._theta_shapes = None
        # Task embedding weights + theta weights.
        self._num_weights = None
        self._num_outputs = None
        # If an external input is required, this may not be None.
        self._size_ext_input = None
        self._target_shapes = None

    def _is_properly_setup(self):
        """This method can be used by classes that implement this interface to
        check whether all required properties have been set."""
        #assert(self._theta is not None)
        #assert(self._task_embs is not None)
        assert(self._theta_shapes is not None)
        assert(self._num_weights is not None)
        assert(self._num_outputs is not None)
        assert(self._target_shapes is not None)

    @property
    def theta(self):
        return self._theta

    @property
    def num_outputs(self):
        """Getter for the attribute num_outputs."""
        return self._num_outputs

    @property
    def num_weights(self):
        """Getter for read-only attribute num_weights."""
        return self._num_weights

    @property
    def has_theta(self):
        """Getter for read-only attribute has_theta."""
        return self._theta is not None

    @property
    def theta_shapes(self):
        """Getter for read-only attribute theta_shapes.

        Returns:
            A list of lists of integers.
        """
        return self._theta_shapes

    @property
    def has_task_embs(self):
        """Getter for read-only attribute has_task_embs."""
        return self._task_embs is not None

    @property
    def num_task_embs(self):
        """Getter for read-only attribute num_task_embs."""
        assert(self.has_task_embs)
        return len(self._task_embs)

    @property
    def requires_ext_input(self):
        """Getter for read-only attribute requires_ext_input."""
        return self._size_ext_input is not None

    @property
    def target_shapes(self):
        """Getter for read-only attribute target_shapes.

        Returns:
            A list of lists of integers.
        """
        return self._target_shapes

    def get_task_embs(self):
        """Return a list of all task embeddings.

        Returns:
            A list of Parameter tensors.
        """
        assert(self.has_task_embs)
        return self._task_embs

    def get_task_emb(self, task_id):
        """Return the task embedding corresponding to a given task id.

        Args:
            task_id: Determines the task for which the embedding should be
                returned.

        Returns:
            A list of Parameter tensors.
        """
        #print('def get_task_emb(self, task_id): in module wrappers')
        assert(self.has_task_embs)
        return self._task_embs[task_id]

    @abstractmethod
    def forward(self, task_id=None, theta=None, dTheta=None, task_emb=None,
                ext_inputs=None, squeeze=True):
        """Compute all HyperWeights.

        Args:
            task_id: The index of the task for which the network should
                produce weights. The corresponding internal task embedding will
                be selected as input. Only one integer can be given!
            theta: List of weight tensors, that are used as network parameters.
                If "has_theta" is False, then this parameter is mandatory.
                Note, when provided, internal parameters (theta) are not used.
            dTheta: List of weight tensors, that are added to "theta" (the
                internal list of parameters or the one given via the option
                "theta"), when computing the output of this network.
            task_emb: If "has_task_embs" is False, then one has to provide the
                task embedding as additional input via this option.
            ext_inputs: If "requires_ext_input" is True, then one has to provide
                the additional embeddings as input here. Note, one might provide
                a batch of embeddings (see option "squeeze" for details).
            squeeze: If a batch of inputs is given, the first dimension of the
                resulting weight tensors will have as first dimension the batch
                dimension. Though, the main network expects this dimension to
                be squeezed away. This will be done automatically if this
                option is enabled (hence, it only has an effect for a batch
                size of 1).

        Returns:
            A list of weights. Two consecutive entries always correspond to a
            weight matrix followed by a bias vector.
        """
        #print('def forward() in module wrappers hyperinterfacess')
        pass # TODO implement

class MainNetInterface(ABC):
    """A general interface for main networks, that can be used stand-alone
    (i.e., having their own weights) or with no (or only some) internal
    weights, such that the remaining weights have to be passed through the
    forward function (e.g., they may be generated through a hypernetwork).

    .. deprecated:: 1.0
        Please use module :class:`mnets.mnet_interface.MainNetInterface`
        instead.

    Attributes:
        weights: A list of all internal weights of the main network. If all
            weights are assumed to be generated externally, then this
            attribute will be None.
        param_shapes: A list of list of integers. Each list represents the
            shape of a parameter tensor. Note, this attribute is
            independent of the attribute "weights", it always comprises the
            shapes of all weight tensors as if the network would be stand-
            alone (i.e., no weights being passed to the forward function).
        hyper_shapes: A list of list of integers. Each list represents the
            shape of a weight tensor that has to be passed to the forward
            function. If all weights are maintained internally, then this
            attribute will be None.
        has_bias: Whether layers in this network have bias terms.
        has_fc_out: Whether the output layer of the network is a fully-
            connected layer.
            Note, if this attribute is set to True, it is implicitly assumed
            that if "hyper_shapes" is not None, the last two entries of
            "hyper_shapes" are the weights and biases of this layer.
        num_params: The total number of weights in the parameter tensors
            described by the attribute "param_shapes".
    """
    def __init__(self):
        """Initialize the network.

        Args:

        """
        super(MainNetInterface, self).__init__()

        warn('Please use class "mnets.mnet_interface.MainNetInterface" ' +
             'instead.', DeprecationWarning)
        #print('class MainNetInterface() in  module wrappers')

        # The following member variables have to be set by all classes that
        # implement this interface.
        self._weights = None
        self._all_shapes = None
        self._hyper_shapes = None
        self._num_params = None
        self._has_bias = None
        self._has_fc_out = None

    def _is_properly_setup(self):
        """This method can be used by classes that implement this interface to
        check whether all required properties have been set."""
        assert(self._weights is not None or self._hyper_shapes is not None)
        if self._weights is not None and self._hyper_shapes is not None:
            assert((len(self._weights) + len(self._hyper_shapes)) == \
                   len(self._all_shapes))
        elif self._weights is not None:
            assert(len(self._weights) == len(self._all_shapes))
        else:
            assert(len(self._hyper_shapes) == len(self._all_shapes))
        assert(self._all_shapes is not None)
        assert(isinstance(self._has_bias, bool))
        assert(isinstance(self._has_fc_out, bool))

    @property
    def weights(self):
        """Getter for read-only attribute weights.

        Returns:
            A :class:`torch.nn.ParameterList` or None, if no parameters are
            internally maintained.
        """
        return self._weights

    @property
    def param_shapes(self):
        """Getter for read-only attribute param_shapes.

        Returns:
            A list of lists of integers.
        """
        return self._all_shapes

    @property
    def hyper_shapes(self):
        """Getter for read-only attribute hyper_shapes.

        Returns:
            A list of lists of integers.
        """
        return self._hyper_shapes

    @property
    def has_bias(self):
        """Getter for read-only attribute has_bias."""
        return self._has_bias

    @property
    def has_fc_out(self):
        """Getter for read-only attribute has_fc_out."""
        return self._has_fc_out

    @property
    def num_params(self):
        """Getter for read-only attribute num_params.

        Returns:
            Total number of parameters in the network.
        """
        if self._num_params is None:
            self._num_params = int(np.sum([np.prod(l) for l in
                                           self.param_shapes]))
        return self._num_params

if __name__ == '__main__':
    pass


