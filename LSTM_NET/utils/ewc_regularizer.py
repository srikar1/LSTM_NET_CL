import numpy as np
import torch
from torch.nn import functional as F

#computing fisher information matrix.
def fisher_mat(task_id, data, params, device, mnet, hnet=None,
               empirical_fisher=True, online=False, gamma=1., n_max=-1,
               regression=False, time_series=False,
               allowed_outputs=None, custom_forward=None, custom_nll=None, logger=None):
    n_samples = data.num_train_samples
    n_samples = 1
    torch.backends.cudnn.enabled = False
    print('n_samples', n_samples)
    if n_max != -1:
        n_samples = min(n_samples, n_max)
    mnet_mode = mnet.training
    mnet.eval()
    if hnet is not None:
        hnet_mode = hnet.training
        hnet.eval()
    fisher = []
    for p in params:
        fisher.append(torch.zeros_like(p))
        assert p.requires_grad
    data.reset_batch_generator(train=True, test=False, val=False)
    for s in range(n_samples):
        if s % 100 == 0:
            print('Training step of fishercomputing: %d ...' % s)
        batch = data.next_train_batch(1)
        X = data.input_to_torch_tensor(batch[0], device, mode='inference')
        T = data.output_to_torch_tensor(batch[1], device, mode='inference')
        if custom_forward is None:
            # weights = hnet.forward(task_id, theta=params)
            weights = hnet.forward(task_id)
            # weights.retain_grad(
            # print(weights.requires_grad)
            Y = mnet.forward(X, weights=weights)
        assert (len(Y.shape) == 2)
        if data.is_one_hot:
            T = torch.argmax(T)
        if empirical_fisher:
            nll = F.nll_loss(F.log_softmax(Y, dim=1), torch.tensor([T]).to(device))
        mnet.zero_grad()
        if hnet is not None:
            hnet.zero_grad()
        params = weights
        for i, p in enumerate(params):
            p.retain_grad()
        torch.autograd.backward(nll, retain_graph=False, create_graph=False)
        for i, p in enumerate(params):
            fisher[i] += torch.pow(p.grad.detach(), 2)
        # print('')
    for i in range(len(params)):
        fisher[i] /= n_samples
    net = mnet
    if hnet is not None:
        net = hnet
    fis = []
    for i, p in enumerate(params):
        buff_w_name, buff_f_name = _ewc_buffer_names(task_id, i, online)
        net.register_buffer(buff_w_name, p.detach().clone())
        if online and task_id > 0:
            prev_fisher_est = getattr(net, buff_f_name)
            # Decay of previous fisher.
            fisher[i] += gamma * prev_fisher_est
        net.register_buffer(buff_f_name, fisher[i].detach().clone())
        fis.append(fisher[i].detach().clone())
    mnet.train(mode=mnet_mode)
    if hnet is not None:
        hnet.train(mode=hnet_mode)
    # print('fisher',len(fisher),len(fisher[4]))
    return fis


def fisher_mat_replay_mnist(task_id, data, params, device, mnet, hnet=None,
                            empirical_fisher=True, online=False, gamma=1., n_max=-1,
                            regression=False, time_series=False,
                            allowed_outputs=None, custom_forward=None, custom_nll=None, logger=None):
    n_samples = data.num_train_samples
    n_samples = 1000
    torch.backends.cudnn.enabled = False
    print('n_samples', n_samples)
    if n_max != -1:
        n_samples = min(n_samples, n_max)
    mnet_mode = mnet.training
    mnet.eval()
    if hnet is not None:
        hnet_mode = hnet.training
        hnet.eval()
    fisher = []
    for p in params:
        fisher.append(torch.zeros_like(p))
        assert p.requires_grad
    data.reset_batch_generator(train=True, test=False, val=False)
    for s in range(n_samples):
        if s % 100 == 0:
            print('Training step of fishercomputing: %d ...' % s)
        batch = data.next_train_batch(1)
        X_real = data.input_to_torch_tensor(batch[0], device, mode='inference')
        T = data.output_to_torch_tensor(batch[1], device, mode='inference')

        mu_var = enc.forward(X_real)
        mu = mu_var[:, 0: config.latent_dim]
        logvar = mu_var[:, config.latent_dim:2 * config.latent_dim]
        kld = compute_kld(mu, logvar, config, t)
        dec_input = reparameterize(mu, logvar)
        reconstructions, weigh = sample(dec, d_hnet, config, t, device, z=dec_input)
        params = weigh

        # x_rec_loss = F.binary_cross_entropy(reconstructions,X_real, reduction='none')
        nll = F.nll_loss(reconstructions, X_real.to(device))
        x_rec_loss = torch.mean(nll, dim=1)
        x_rec_loss = torch.mean(x_rec_loss)

        loss = x_rec_loss + kld

        ######################################################
        # HYPERNET REGULARISATION - CONTINUAL LEARNING METHOD
        ######################################################

        # loss.backward(retain_graph=calc_reg, create_graph=calc_reg and   config.backprop_dt)
        torch.autograd.backward(loss, retain_graph=False, create_graph=False)
        for i, p in enumerate(params):
            # print('i,p', i, p)
            # print('p.grad', p.grad)
            fisher[i] += torch.pow(p.grad.detach(), 2)
            # assert p.grad.shape == p.shape
    for i in range(len(params)):
        fisher[i] /= n_samples
    net = mnet
    if hnet is not None:
        net = hnet
    fis = []
    for i, p in enumerate(params):
        buff_w_name, buff_f_name = _ewc_buffer_names(task_id, i, online)
        net.register_buffer(buff_w_name, p.detach().clone())
        if online and task_id > 0:
            prev_fisher_est = getattr(net, buff_f_name)
            # Decay of previous fisher.
            fisher[i] += gamma * prev_fisher_est
        net.register_buffer(buff_f_name, fisher[i].detach().clone())
        fis.append(fisher[i].detach().clone())
    mnet.train(mode=mnet_mode)
    if hnet is not None:
        hnet.train(mode=hnet_mode)
    return fis


def fisher_mat_ewc(task_id, data, params, device, mnet, hnet=None,
                   empirical_fisher=True, online=False, gamma=1., n_max=-1,
                   regression=False, time_series=False,
                   allowed_outputs=None, custom_forward=None, custom_nll=None, logger=None):
    n_samples = data.num_train_samples
    n_samples = 100

    torch.backends.cudnn.enabled = False
    print('fisher n_samples', n_samples)
    if n_max != -1:
        n_samples = min(n_samples, n_max)
    # print('number of sample', n_samples)
    mnet_mode = mnet.training
    mnet.eval()
    if hnet is not None:
        hnet_mode = hnet.training
        hnet.eval()
    fisher = []
    for p in params:
        # p.requires_grad = True
        fisher.append(torch.zeros_like(p))
        # assert p.requires_grad
    print('len(params', len(params))
    data.reset_batch_generator(train=True, test=False, val=False)
    for s in range(n_samples):
        # if s % 1000 == 0:
        # logger.info('Training step of fisher computing: %d ...' % s)
        batch = data.next_train_batch(1)
        X = data.input_to_torch_tensor(batch[0], device, mode='inference')
        T = data.output_to_torch_tensor(batch[1], device, mode='inference')
        if custom_forward is None:
            weights = hnet.forward(task_id, theta=params)
            Y = mnet.forward(X, weights=weights)

        # params = weights
        assert (len(Y.shape) == 2)
        if data.is_one_hot:
            T = torch.argmax(T)
        if empirical_fisher:
            nll = F.nll_loss(F.log_softmax(Y, dim=1),
                             torch.tensor([T]).to(device))
        mnet.zero_grad()
        if hnet is not None:
            hnet.zero_grad()

        # for i, p in enumerate(params):
        #   p.retain_grad()
        torch.autograd.backward(nll, retain_graph=False, create_graph=False)
        # print('params', params)
        for i, p in enumerate(params):
            # print('p.grad', p.requires_grad)
            fisher[i] += torch.pow(p.grad.detach(), 2)

    for i in range(len(params)):
        fisher[i] /= n_samples
    net = mnet
    if hnet is not None:
        net = hnet
    fis = []
    for i, p in enumerate(params):
        buff_w_name, buff_f_name = _ewc_buffer_names(task_id, i, online)
        net.register_buffer(buff_w_name, p.detach().clone())
        if online and task_id > 0:
            prev_fisher_est = getattr(net, buff_f_name)

            # Decay of previous fisher.
            fisher[i] += gamma * prev_fisher_est

        net.register_buffer(buff_f_name, fisher[i].detach().clone())
        fis.append(fisher[i].detach().clone())
    mnet.train(mode=mnet_mode)
    if hnet is not None:
        hnet.train(mode=hnet_mode)
    return fis


def ewc_regularizer(task_id, params, mnet, hnet=None, online=False, gamma=1.):
    assert (task_id > 0)
    net = mnet
    if hnet is not None:
        net = hnet
    ewc_reg = 0
    num_prev_tasks = 1 if online else task_id
    # print('num_prev_tasks,online',num_prev_tasks,online)
    for t in range(num_prev_tasks):
        for i, p in enumerate(params):
            if i > 16:
                buff_w_name, buff_f_name = _ewc_buffer_names(t, i, online)
                prev_weights = getattr(net, buff_w_name)
                fisher_est = getattr(net, buff_f_name)
                if online:
                    fisher_est *= gamma
                # ewc_reg += (fisher_est * (p - prev_weights).pow(2)).sum()
                ewc_reg += ((p - prev_weights).pow(2)).sum()


    # print('ewc_reg',ewc_reg)
    return ewc_reg


def _ewc_buffer_names(task_id, param_id, online):
    """
    Returns:
        (tuple): Tuple containing:
        - **weight_buffer_name**
        - **fisher_estimate_buffer_name**
    """
    task_ident = '' if online else '_task_%d' % task_id
    weight_name = 'ewc_prev{}_weights_{}'.format(task_ident, param_id)
    fisher_name = 'ewc_fisher_estimate{}_weights_{}'.format(task_ident, param_id)
    return weight_name, fisher_name


def context_mod_forward(mod_weights=None):
    """Create a custom forward function for function :func:`compute_fisher`.

    See argument ``custom_forward`` of function :func:`compute_fisher` for more
    details.

    This is a helper method to quickly retrieve a function handle that manages
    the forward pass for a context-modulated main network.

    We assume that the interface of the main network is similar to the one of
    :meth:`mnets.mlp.MLP.forward`.

    Args:
        mod_weights (optional): If provided, it is assumed that
            :func:`compute_fisher` is called with ``hnet`` set to ``None``.
            Hence, the returned function handle will have the given
            context-modulation pattern hard-coded.
            If left unspecified, it is assumed that a ``hnet`` is passed to
            :func:`compute_fisher` and that this ``hnet`` computes only the
            parameters of all context-mod layers.

    Returns:
        A function handle.
    """

    def hnet_forward(mnet, hnet, task_id, params, X):
        mod_weights = hnet.forward(task_id)
        weights = {
            'mod_weights': mod_weights,
            'internal_weights': params
        }
        Y = mnet.forward(X, weights=weights)
        return Y

    def mnet_only_forward(mnet, params, X):
        weights = {
            'mod_weights': mod_weights,
            'internal_weights': params
        }
        Y = mnet.forward(X, weights=weights)
        return Y

    if mod_weights is None:
        return hnet_forward
    else:
        return mnet_only_forward


def cognet_mse_nll(no_fix_unit_amplification=False):
    r"""Create a custom NLL function for function
    :func:`utils.ewc_regularizer.compute_fisher`.

    Here, we consider a set of cognitive tasks as suggested by

        https://www.nature.com/articles/s41593-018-0310-2

    We assume the network loss is computed as described in section *Training
    procedure* on pg. 12 of the paper lined above.

    Thus the network has an output shape of ``[S, N, F]``, where ``S`` is the
    length of a time sequence, ``N`` is the batch size (we can assume ``N`` is 1
    in function :func:`utils.ewc_regularizer.compute_fisher`) and ``F`` is the
    number of output classes. The network is trained using masked MSE loss.

    Note, there are 9 output classes (8 *output ring units* and 1 *fixation
    output unit*), where the last class (the *fixation output unit*) might be
    treated differently.

    The first 10 timesteps (100 ms) are ignored. The fixation period is defined
    by the timesteps that are associated with label 8.

    During the fixation period, the MSE between ring units and targets
    (which will be zero) will be weighted by 1, whereas the MSE between the
    fixation unit and its target (which will be 1) is weighted by 2.

    During the response period, the weighting will change to 5 for ring units
    and 10 for fixation units.

    Similar to function :func:`utils.ewc_regularizer.compute_fisher` (cmp.
    argument ``time_series``), we adopt the following decomposition of the joint

    .. math::

        p(\mathbf{y} \mid \theta; \mathbf{x}) =
        \prod_{i=1}^S p(\mathbf{y}_i \mid \mathbf{y}_1, \dots,
        \mathbf{y}_{i-1}, \theta; \mathbf{x}_i)

    Since the loss is a masked MSE loss, we assume the predictive distribution
    per time step is a Gaussian
    :math:`\mathcal{N}(\mathbf{\mu}, I \mathbf{\sigma}^2)` with diagonal
    covariance matrix.

    Hence, we can write the NLL for the :math:`i`-th output as follows, assuming
    :math:`\textbf{t}_i` is the corresponding 1-hot target:

    .. math::

        \text{NLL}_i &= - \log p(\mathbf{y}_i = \mathbf{t}_i
        \mid \mathbf{y}_1, \dots, \mathbf{y}_{i-1}, \theta; \mathbf{x}_i)\\
        &= \text{const.} + \frac{1}{2} \sum_{j=0}^8 \frac{1}{\sigma_{i,j}^2}
        \big(f(\mathbf{x}_i, \mathbf{h}_{i-1}, \theta)_j - t_{i,j} \big)^2\\
        &= \text{const.} + \sum_{j=0}^8 \tau_{i,j}
        \big(f(\mathbf{x}_i, \mathbf{h}_{i-1}, \theta)_j - t_{i,j} \big)^2

    where we defined :math:`\tau_{i,j} \equiv \frac{1}{2 \sigma_{i,j}^2}`.
    Based on the time step :math:`i` and the output unit :math:`j`, we can set
    the variance such that :math:`\tau_{i,j}` corresponds to the masking value
    as defined above.

    The complete NLL over all timesteps is then simply:

    .. math::

        \text{NLL} &= \sum_{i=1}^S \text{NLL}_i \\
        &= \text{const.} + \sum_{i=1}^S \sum_{j=0}^8 \tau_{i,j}
        \big(f(\mathbf{x}_i, \mathbf{h}_{i-1}, \theta)_j - t_{i,j} \big)^2

    Note, a mask value of zero (:math:`\tau_{i,j} = 0`) corresponds to infinite
    variance.

    Args:
        no_fix_unit_amplification (bool): If ``True``, then the masking for
            the fixation unit is not amplified (by a factor of 2) as described
            above. Instead, fixation and ring units are treated equally.
    """

    def custom_nll(Y, T, data, allowed_outputs, empirical_fisher):
        # We expect targets to be given as 1-hot encodings.
        assert (np.all(np.equal(list(Y.shape), list(T.shape))))

        # Fixation period is defined by timesteps having label 8.
        labels = torch.argmax(T, 2)

        if no_fix_unit_amplification:
            mask = torch.ones(T.shape[0], T.shape[1])
            mask[labels != 8] = 5
            mask[0:10, :] = 0

            # Make sure that `mask` are broadcastable wrt `Y` and `T`.
            mask = mask.view(T.shape[0], T.shape[1], 1)

        else:
            mask = torch.ones_like(T)
            # Make sure that `labels` can be used to index `mask`.
            labels = labels.view(T.shape[0], T.shape[1], 1)
            labels = labels.expand(mask.size())

            mask[labels != 8] = 5
            mask[0:10, :, :] = 0

            mask[:, :, 8] = 2 * mask[:, :, 8]

        nll = (mask * (Y - T) ** 2).sum()

        return nll

    return custom_nll


if __name__ == '__main__':
    pass
