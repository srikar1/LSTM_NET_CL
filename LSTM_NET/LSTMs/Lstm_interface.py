from abc import ABC, abstractmethod
import numpy as np
from warnings import warn

class LstmHyperNetInterface(ABC):
    def __init__(self):
        """Initialize the network."""
        super(LstmHyperNetInterface, self).__init__()
        print('LstmHyperNetInterface')
        self._theta = None
        self._task_embs = None
        self._theta_shapes = None
        # Task embedding weights + theta weights.
        self._num_weights = None
        self._num_outputs = None
        # If an external input is required, this may not be None.
        self._size_ext_input = None
        self._target_shapes = None
        self.fa=None

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
        """Getter for read-only attribute theta.

        Theta are all learnable parameters of the hypernet except the task
        embeddings, i.e., theta comprises all parameters that should be
        regularized in order to avoid catastrophic forgetting when training
        the hypernetwork in a Continual Learning setting.

        Returns:
            A :class:`torch.nn.ParameterList` or None, if this network has no
            weights.
        """
        #print('def theta(self): in module wrappers')
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

        return self._size_ext_input is not None

    @property
    def target_shapes(self):

        return self._target_shapes

    def get_task_embs(self):

        assert(self.has_task_embs)
        return self._task_embs

    def get_task_emb(self, task_id):

        #print('def get_task_emb(self, task_id): in module wrappers')
        assert(self.has_task_embs)
        return self._task_embs[task_id]

    @abstractmethod
    def forward(self, task_id=None, theta=None, dTheta=None, task_emb=None,
                ext_inputs=None, squeeze=True):

        #print('def forward() in LstmHyperNetInterface')
        pass # TODO implement