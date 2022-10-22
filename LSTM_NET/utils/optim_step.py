from torch import optim
import torch
import math

def calc_delta_theta(optimizer, use_sgd_change, lr=None, detach_dt=True):
    r"""Calculate :math:`\Delta\theta`, i.e., the change in trainable parameters
    (:math:`\theta`) in order to minimize the task-specific loss.

    **Note**, one has to call :func:`torch.autograd.backward` on a
    desired loss before calling this function, otherwise there are no gradients
    to compute the weight change that the optimizer would cause. Hence, this
    method is called in between :func:`torch.autograd.backward` and
    :meth:`torch.optim.Optimizer.step`.

    Note, by default, gradients are detached from the computational graph.

    Args:
        optimizer: The optimizer that will be used to change :math:`\theta`.
        use_sgd_change: If :code:`True`, then we won't calculate the actual step
            done by the current optimizer, but the one that would be done by a
            simple SGD optimizer.
        lr: Has to be specified if `use_sgd_change` is :code:`True`. The
            learning rate if the optimizer.
        detach_dt: Whether :math:`\Delta\theta` should be detached from the
            computational graph. Note, in order to backprop through
            :math:`\Delta\theta`, you have to call
            :func:`torch.autograd.backward` with `create_graph` set to
            :code:`True` before calling this method.

    Returns:
        :math:`\Delta\theta`
    """
    assert(not use_sgd_change or lr is not None)

    if use_sgd_change:
        ret = []
        for g in optimizer.param_groups:
            for p in g['params']:
                if detach_dt:
                    ret.append(-lr * p.grad.detach().clone())
                else:
                    ret.append(-lr * p.grad.clone())
        return ret
    else:
        if isinstance(optimizer, optim.SGD):
            return sgd_step(optimizer, detach_dp=detach_dt)
        if isinstance(optimizer, optim.Adam):
            return adam_step(optimizer, detach_dp=detach_dt)
        if isinstance(optimizer, optim.RMSprop):
            return rmsprop_step(optimizer, detach_dp=detach_dt)
        else:
            raise NotImplementedError('Not implemented for optimizer %s' %
                                      optimizer.type)

def sgd_step(optimizer, detach_dp=True):
    """Performs a single optimization step using the SGD optimizer. The code
    has been copied from:

        https://git.io/fjYit

    Note, this function does not change the inner state of the given
    optimizer object.

    Note, gradients are cloned and detached by default.

    Args:
        optimizer: An instance of class :class:`torch.optim.SGD`.
        detach_dp: Whether gradients are detached from the computational
            graph. Note, :code:`False` only makes sense if
            func:`torch.autograd.backward` was called with the argument
            `create_graph` set to :code:`True`.

    Returns:
        A list of gradient changes `d_p` that would be applied by this
        optimizer to all parameters when calling :meth:`torch.optim.SGD.step`.
    """
    assert(isinstance(optimizer, optim.SGD))

    d_ps = []

    for group in optimizer.param_groups:
        weight_decay = group['weight_decay']
        momentum = group['momentum']
        dampening = group['dampening']
        nesterov = group['nesterov']

        for p in group['params']:
            if p.grad is None:
                continue

            if detach_dp:
                d_p = p.grad.detach().clone()
            else:
                d_p = p.grad.clone()

            if weight_decay != 0:
                d_p.add_(weight_decay, p.data)
            if momentum != 0:
                orig_state = dict(optimizer.state[p])
                param_state = dict()

                if 'momentum_buffer' in orig_state:
                    param_state['momentum_buffer'] = \
                        orig_state['momentum_buffer'].clone()

                if 'momentum_buffer' not in param_state:
                    buf = torch.clone(d_p).detach()
                else:
                    buf = param_state['momentum_buffer']
                    buf.mul_(momentum).add_(1 - dampening, d_p)
                    #buf = buf.mul(momentum).add(1 - dampening, d_p)
                if nesterov:
                    d_p = d_p.add(momentum, buf)
                else:
                    d_p = buf

            d_ps.append(-group['lr'] * d_p)

    return d_ps

def adam_step(optimizer, detach_dp=True):
    """Performs a single optimization step using the Adam optimizer. The code
    has been copied from:

        https://git.io/fjYP3

    Note, this function does not change the inner state of the given
    optimizer object.

    Note, gradients are cloned and detached by default.

    Args:
        optimizer: An instance of class :class:`torch.optim.Adam`.
        detach_dp: Whether gradients are detached from the computational
            graph. Note, :code:`False` only makes sense if
            func:`torch.autograd.backward` was called with the argument
            `create_graph` set to :code:`True`.

    Returns:
        A list of gradient changes `d_p` that would be applied by this
        optimizer to all parameters when calling :meth:`torch.optim.Adam.step`.
    """
    assert (isinstance(optimizer, optim.Adam))

    d_ps = []

    for group in optimizer.param_groups:
        for p in group['params']:
            if p.grad is None:
                continue

            if detach_dp:
                grad = p.grad.detach().clone()
            else:
                grad = p.grad.clone()

            if grad.is_sparse:
                raise RuntimeError(
                    'Adam does not support sparse gradients, please consider SparseAdam instead')
            amsgrad = group['amsgrad']
            if amsgrad and not detach_dp:
                raise ValueError('Cannot backprop through optimizer step if ' +
                                 '"amsgrad" is enabled.')

            orig_state = dict(optimizer.state[p])
            state = dict()

            # State initialization
            if len(orig_state) == 0:
                orig_state['step'] = 0
                # Exponential moving average of gradient values
                orig_state['exp_avg'] = torch.zeros_like(p.data)
                # Exponential moving average of squared gradient values
                orig_state['exp_avg_sq'] = torch.zeros_like(p.data)
                if amsgrad:
                    # Maintains max of all exp. moving avg. of sq. grad. values
                    orig_state['max_exp_avg_sq'] = torch.zeros_like(p.data)

            # Copy original state.
            state['step'] = int(orig_state['step'])
            state['exp_avg'] = orig_state['exp_avg'].clone()
            state['exp_avg_sq'] = orig_state['exp_avg_sq'].clone()
            if amsgrad:
                state['max_exp_avg_sq'] = orig_state['max_exp_avg_sq'].clone()

            exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
            if amsgrad:
                max_exp_avg_sq = state['max_exp_avg_sq']
            beta1, beta2 = group['betas']

            state['step'] += 1

            if group['weight_decay'] != 0:
                #grad.add_(group['weight_decay'], p.data)
                grad.add(group['weight_decay'], p.data)

            # Decay the first and second moment running average coefficient
            exp_avg.mul_(beta1).add_(1 - beta1, grad)
            #exp_avg.mul_(beta1)
            #exp_avg += (1 - beta1) * grad

            exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
            #exp_avg_sq.mul_(beta2)
            #exp_avg_sq = torch.addcmul(exp_avg_sq, 1 - beta2, grad, grad)
            if amsgrad:
                # Maintains the maximum of all 2nd moment running avg. till now
                torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                # Use the max. for normalizing running avg. of gradient
                denom = max_exp_avg_sq.sqrt().add_(group['eps'])
            else:
                #denom = exp_avg_sq.sqrt().add_(group['eps'])
                denom = exp_avg_sq.sqrt() + group['eps']

            bias_correction1 = 1 - beta1 ** state['step']
            bias_correction2 = 1 - beta2 ** state['step']
            step_size = group['lr'] * math.sqrt(
                bias_correction2) / bias_correction1

            d_ps.append(-step_size * (exp_avg / denom))

    return d_ps

def rmsprop_step(optimizer, detach_dp=True):
    """Performs a single optimization step using the RMSprop optimizer. The code
    has been copied from:

        https://git.io/fjurp

    Note, this function does not change the inner state of the given
    optimizer object.

    Note, gradients are cloned and detached by default.

    Args:
        optimizer: An instance of class :class:`torch.optim.Adam`.
        detach_dp: Whether gradients are detached from the computational
            graph. Note, :code:`False` only makes sense if
            func:`torch.autograd.backward` was called with the argument
            `create_graph` set to :code:`True`.

    Returns:
        A list of gradient changes `d_p` that would be applied by this
        optimizer to all parameters when calling
        :meth:`torch.optim.RMSprop.step`.
    """
    assert (isinstance(optimizer, optim.RMSprop))

    d_ps = []

    for group in optimizer.param_groups:
        for p in group['params']:
            if p.grad is None:
                continue

            if detach_dp:
                grad = p.grad.detach().clone()
            else:
                grad = p.grad.clone()

            if grad.is_sparse:
                raise RuntimeError('RMSprop does not support sparse gradients')

            orig_state = dict(optimizer.state[p])
            state = dict()

            # State initialization
            if len(orig_state) == 0:
                orig_state['step'] = 0
                orig_state['square_avg'] = torch.zeros_like(p.data)
                if group['momentum'] > 0:
                    orig_state['momentum_buffer'] = torch.zeros_like(p.data)
                if group['centered']:
                    orig_state['grad_avg'] = torch.zeros_like(p.data)

            # Copy original state.
            state['step'] = int(orig_state['step'])
            state['square_avg'] = orig_state['square_avg'].clone()
            if group['momentum'] > 0:
                state['momentum_buffer'] = orig_state['momentum_buffer'].clone()
            if group['centered']:
                state['grad_avg'] = orig_state['grad_avg'].clone()

            square_avg = state['square_avg']
            alpha = group['alpha']

            state['step'] += 1

            if group['weight_decay'] != 0:
                grad = grad.add(group['weight_decay'], p.data)

            #square_avg.mul_(alpha).addcmul_(1 - alpha, grad, grad)
            square_avg = square_avg.mul(alpha).addcmul(1 - alpha, grad, grad)

            if group['centered']:
                grad_avg = state['grad_avg']
                grad_avg.mul_(alpha).add_(1 - alpha, grad)
                #avg = square_avg.addcmul(-1, grad_avg, grad_avg).sqrt().add_(group['eps'])
                avg = square_avg.addcmul(-1, grad_avg, grad_avg).sqrt().\
                    add(group['eps'])
            else:
                #avg = square_avg.sqrt().add_(group['eps'])
                avg = square_avg.sqrt().add(group['eps'])

            if group['momentum'] > 0:
                buf = state['momentum_buffer']
                #buf.mul_(group['momentum']).addcdiv_(grad, avg)
                buf = buf.mul(group['momentum']) + grad / avg

                d_ps.append(-group['lr'] * buf)
            else:
                d_ps.append(-group['lr'] * (grad / avg))

    return d_ps

if __name__ == '__main__':
    pass

