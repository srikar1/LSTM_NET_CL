from argparse import Namespace
import torch
import torch.optim as optim
from torch.nn import functional as F
import numpy as np
from time import time

from cifar import train_utils as tutils
from mnets.classifier_interface import Classifier
from utils import sim_utils as sutils
import utils.optim_step as opstep
import utils.hnet_regularizer as hreg
from utils.torch_utils import get_optimizer
import utils.ewc_regularizer as ewc

# most of the content taken from continual learning with hypernetwork paper.
def test(task_id, data, mnet, hnet, device, shared, config, writer, logger, theta_saved=None,
         train_iter=None, task_emb=None, cl_scenario=None, test_size=None):
    """Evaluate the current performance using the test set.

    Note:
        The hypernetwork ``hnet`` may be ``None``, in which case it is assumed
        that the main network ``mnet`` has internal weights.

    Args:
        (....): See docstring of function :func:`train`.
        train_iter (int, optional): The current training iteration. If given, it
            is used for tensorboard logging.
        task_emb (torch.Tensor, optional): Task embedding. If given, no task ID
            will be provided to the hypernetwork. This might be useful if the
            performance of other than the trained task embeddings should be
            tested.

            .. note::
                This option may only be used for ``cl_scenario=1``. It doesn't
                make sense if the task ID has to be inferred.
        cl_scenario (int, optional): In case the system should be tested on
            another CL scenario than the one user-defined in ``config``.
            
            .. note::
                It is up to the user to ensure that the CL scnearios are
                compatible in this implementation.
        test_size (int, optional): In case the testing shouldn't be performed
            on the entire test set, this option can be used to specify the
            number of test samples to be used.

    Returns:
        (tuple): Tuple containing:

        - **test_acc**: Test accuracy on classification task.
        - **task_acc**: Task prediction accuracy (always 100% for **CL1**).
    """
    if cl_scenario is None:
        cl_scenario = config.cl_scenario
    else:
        assert cl_scenario in [1, 2, 3]

    # `task_emb` ignored for other cl scenarios!
    assert task_emb is None or cl_scenario == 1, \
        '"task_emb" may only be specified for CL1, as we infer the ' + \
        'embedding for other scenarios.'

    mnet.eval()
    if hnet is not None:
        hnet.eval()

    if train_iter is None:
        logger.info('### Test run ...')
    else:
        logger.info('# Testing network before running training step %d ...' % \
                    train_iter)

    # We need to tell the main network, which batch statistics to use, in case
    # batchnorm is used and we checkpoint the batchnorm stats.
    mnet_kwargs = {}
    if mnet.batchnorm_layers is not None:
        if config.bn_distill_stats:
            raise NotImplementedError()
        elif not config.bn_no_running_stats and \
                not config.bn_no_stats_checkpointing:
            # Specify current task as condition to select correct
            # running stats.
            mnet_kwargs['condition'] = task_id

            if task_emb is not None:
                # NOTE `task_emb` might have nothing to do with `task_id`.
                logger.warning('Using batch statistics accumulated for task ' +
                               '%d for batchnorm, but testing is ' % task_id +
                               'performed using a given task embedding.')

    with torch.no_grad():
        batch_size = config.val_batch_size
        # FIXME Assuming all output heads have the same size.
        n_head = data.num_classes

        if test_size is None or test_size >= data.num_test_samples:
            test_size = data.num_test_samples
        else:
            # Make sure that we always use the same test samples.
            data.reset_batch_generator(train=False, test=True, val=False)
            logger.info('Note, only part of test set is used for this test ' +
                        'run!')

        test_loss = 0.0

        # We store all predicted labels and tasks while going over individual
        # test batches.
        correct_labels = np.empty(test_size, np.int)
        pred_labels = np.empty(test_size, np.int)
        correct_tasks = np.ones(test_size, np.int) * task_id
        pred_tasks = np.empty(test_size, np.int)

        curr_bs = batch_size
        N_processed = 0

        # Sweep through the test set.
        while N_processed < test_size:
            if N_processed + curr_bs > test_size:
                curr_bs = test_size - N_processed
            N_processed += curr_bs

            batch = data.next_test_batch(curr_bs)
            X = data.input_to_torch_tensor(batch[0], device)
            T = data.output_to_torch_tensor(batch[1], device)

            ############################
            ### Get main net weights ###
            ############################
            if hnet is None:
                weights = None
            elif cl_scenario > 1:
                raise NotImplementedError()
            elif task_emb is not None:
                if theta_saved is None:
                    theta_ind = None
                    x = False
                else:
                    theta_ind = theta_saved[task_id]
                    x = True
                weights = hnet.forward(task_emb=task_emb, theta=theta_ind, test=x)
            else:
                if theta_saved is None:
                    theta_ind = None
                    x = False
                else:
                    theta_ind = theta_saved[task_id]
                    x = True
                weights = hnet.forward(task_id=task_id, theta=theta_ind, test=x)

            #######################
            ### Get predictions ###
            #######################
            Y_hat_logits = mnet.forward(X, weights=weights, **mnet_kwargs)

            if config.cl_scenario == 1:
                # Select current head.
                task_out = [task_id * n_head, (task_id + 1) * n_head]
            elif config.cl_scenario == 2:
                # Only 1 output head.
                task_out = [0, n_head]
            else:
                raise NotImplementedError()
                # TODO Choose the predicted output head per sample.
                # task_out = [predicted_task_id[0]*n_head,
                #            (predicted_task_id[0]+1)*n_head]

            Y_hat_logits = Y_hat_logits[:, task_out[0]:task_out[1]]
            # We take the softmax after the output neurons are chosen.
            Y_hat = F.softmax(Y_hat_logits, dim=1).cpu().numpy()

            correct_labels[N_processed - curr_bs:N_processed] = \
                T.argmax(dim=1, keepdim=False).cpu().numpy()

            pred_labels[N_processed - curr_bs:N_processed] = \
                Y_hat.argmax(axis=1)

            # Set task prediction to 100% if we do not infer it.
            if cl_scenario > 1:
                raise NotImplementedError()
                # pred_tasks[N_processed-curr_bs:N_processed] = \
                #    predicted_task_id.cpu().numpy()
            else:
                pred_tasks[N_processed - curr_bs:N_processed] = task_id

            # Note, targets are 1-hot encoded.
            test_loss += Classifier.logit_cross_entropy_loss(Y_hat_logits, T,
                                                             reduction='sum')

        class_n_correct = (correct_labels == pred_labels).sum()
        test_acc = 100.0 * class_n_correct / test_size

        task_n_correct = (correct_tasks == pred_tasks).sum()
        task_acc = 100.0 * task_n_correct / test_size

        test_loss /= test_size

        msg = '### Test accuracy of task %d' % (task_id + 1) \
              + (' (before training iteration %d)' % train_iter if \
                     train_iter is not None else '') \
              + ': %.3f' % (test_acc) \
              + (' (using a given task embedding)' if task_emb is not None \
                     else '') \
              + (' - task prediction accuracy: %.3f' % task_acc if \
                     cl_scenario > 1 else '')
        logger.info(msg)

        if train_iter is not None:
            writer.add_scalar('test/task_%d/class_accuracy' % task_id,
                              test_acc, train_iter)

            if config.cl_scenario > 1:
                writer.add_scalar('test/task_%d/task_pred_accuracy' % \
                                  task_id, task_acc, train_iter)

        return test_acc, task_acc


def train(task_id, data, mnet, hnet, device, config, shared, writer, logger, fisher, prev_weights_fish):
    start_time = time()
    print('config.beta', config.beta)
    logger.info('Training network ...')
    # print('def train()')
    mnet.train()
    if hnet is not None:
        hnet.train()
    #################
    ### Optimizer ###
    #################
    # Define the optimizers used to train main network and hypernet.
    if hnet is not None:
        theta_params = list(hnet.theta)
        if config.continue_emb_training:
            for i in range(task_id):  # for all previous task embeddings
                theta_params.append(hnet.get_task_emb(i))

        # for i in range(0,6):
        #  print('hnet.get_task_emb(i)',hnet.get_task_emb(i))
        emb_optimizer = get_optimizer([hnet.get_task_emb(task_id)],
                                      config.lr, momentum=config.momentum,
                                      weight_decay=config.weight_decay, use_adam=config.use_adam,
                                      adam_beta1=config.adam_beta1, use_rmsprop=config.use_rmsprop)
    else:
        theta_params = mnet.weights
        emb_optimizer = None

    theta_optimizer = get_optimizer(theta_params, config.lr,
                                    momentum=config.momentum, weight_decay=config.weight_decay,
                                    use_adam=config.use_adam, adam_beta1=config.adam_beta1,
                                    use_rmsprop=config.use_rmsprop)

    ################################
    ### Learning rate schedulers ###
    ################################
    if config.plateau_lr_scheduler:
        assert (config.epochs != -1)
        # The scheduler config has been taken from here:
        # https://keras.io/examples/cifar10_resnet/
        # Note, we use 'max' instead of 'min' as we look at accuracy rather
        # than validation loss!
        plateau_scheduler_theta = optim.lr_scheduler.ReduceLROnPlateau( \
            theta_optimizer, 'max', factor=np.sqrt(0.1), patience=5,
            min_lr=0.5e-6, cooldown=0)
        plateau_scheduler_emb = None
        if emb_optimizer is not None:
            plateau_scheduler_emb = optim.lr_scheduler.ReduceLROnPlateau( \
                emb_optimizer, 'max', factor=np.sqrt(0.1), patience=5,
                min_lr=0.5e-6, cooldown=0)

    if config.lambda_lr_scheduler:
        assert (config.epochs != -1)

        def lambda_lr(epoch):
            lr_scale = 1.
            if epoch > 180:
                lr_scale = 0.5e-3
            elif epoch > 160:
                lr_scale = 1e-3
            elif epoch > 120:
                lr_scale = 1e-2
            elif epoch > 80:
                lr_scale = 1e-1
            return lr_scale

        lambda_scheduler_theta = optim.lr_scheduler.LambdaLR(theta_optimizer,
                                                             lambda_lr)
        lambda_scheduler_emb = None
        if emb_optimizer is not None:
            lambda_scheduler_emb = optim.lr_scheduler.LambdaLR(emb_optimizer,
                                                               lambda_lr)
    ##############################
    ### Prepare CL Regularizer ###
    ##############################
    # Whether we will calculate the regularizer.
    calc_reg = task_id > 0 and not config.mnet_only and config.beta > 0 and not config.train_from_scratch
    regged_outputs = None
    if calc_reg:
        if config.online_target_computation:
            # Compute targets for the regularizer whenever they are needed.
            # -> Computationally expensive.
            # print('if config.online_target_computation is true')
            targets_hypernet = None
            prev_theta = [p.detach().clone() for p in hnet.theta]
            prev_task_embs = [p.detach().clone() for p in hnet.get_task_embs()]
        else:
            # Compute targets for the regularizer once and keep them all in
            # memory -> Memory expensive.
            # print('if config.online_target_computation is false')
            # this works
            targets_hypernet = hreg.get_current_targets(task_id, hnet)
            prev_theta = None
            prev_task_embs = None

        # If we do not want to regularize all outputs (in a multi-head setup).
        # Note, we don't care whether output heads other than the current one
        # change.
        regged_outputs = None
        if config.cl_scenario != 2:
            # FIXME We assume here that all tasks have the same output size.
            n_y = data.num_classes
            print('n_y', n_y)
            regged_outputs = [list(range(i * n_y, (i + 1) * n_y)) for i in
                              range(task_id)]
            print('regged_outputs', regged_outputs)

    # We need to tell the main network, which batch statistics to use, in case
    # batchnorm is used and we checkpoint the batchnorm stats.
    mnet_kwargs = {}
    if mnet.batchnorm_layers is not None:
        if config.bn_distill_stats:
            raise NotImplementedError()
        elif not config.bn_no_running_stats and \
                not config.bn_no_stats_checkpointing:
            # Specify current task as condition to select correct
            # running stats.
            mnet_kwargs['condition'] = task_id

    ######################
    ### Start training ###
    ######################

    iter_per_epoch = -1
    if config.epochs == -1:
        training_iterations = config.n_iter
    else:
        assert (config.epochs > 0)
        iter_per_epoch = int(np.ceil(data.num_train_samples / \
                                     config.batch_size))
        training_iterations = config.epochs * iter_per_epoch

    summed_iter_runtime = 0
    print('data.num_train_samples,config.batch_size,config.epochs', data.num_train_samples, config.batch_size,
          config.epochs)
    print('training_iterations', training_iterations)

    for i in range(training_iterations):
        ### Evaluate network.
        # We test the network before we run the training iteration.
        # That way, we can see the initial performance of the untrained network.
        if i % config.val_iter == 0:
            # test(task_id, data, mnet, hnet, device, shared, config, writer,logger, train_iter=i)
            mnet.train()
            if hnet is not None:
                hnet.train()

        if i % 200 == 0:
            logger.info('Training step: %d ...' % i)

        iter_start_time = time()

        theta_optimizer.zero_grad()
        if emb_optimizer is not None:
            emb_optimizer.zero_grad()

        #######################################
        ### Data for current task and batch ###
        #######################################
        batch = data.next_train_batch(config.batch_size)
        X = data.input_to_torch_tensor(batch[0], device, mode='train')
        T = data.output_to_torch_tensor(batch[1], device, mode='train')

        # Get the output neurons depending on the continual learning scenario.
        n_y = data.num_classes
        if config.cl_scenario == 1:
            # Choose current head.
            task_out = [task_id * n_y, (task_id + 1) * n_y]
        elif config.cl_scenario == 2:
            # Always all output neurons, only one head is used.
            task_out = [0, n_y]
        else:
            # Choose current head, which will be inferred during inference.
            task_out = [task_id * n_y, (task_id + 1) * n_y]

        ########################
        ### Loss computation ###
        ########################
        if config.mnet_only:
            weights = None
        else:
            weights = hnet.forward(task_id=task_id)
            # print('weights.requires_grad', weights.requires_grad)
        Y_hat_logits = mnet.forward(X, weights, **mnet_kwargs)
        # print('weights.requires_grad', weights.requires_grad)
        # print('-----------------')

        # Restrict output neurons
        Y_hat_logits = Y_hat_logits[:, task_out[0]:task_out[1]]
        # print('Y_hat_logits', Y_hat_logits.shape)
        assert (T.shape[1] == Y_hat_logits.shape[1])
        # compute loss on task and compute gradients
        if config.soft_targets:
            soft_label = 0.95
            num_classes = data.num_classes
            soft_targets = torch.where(T == 1,
                                       torch.Tensor([soft_label]),
                                       torch.Tensor([(1 - soft_label) / (num_classes - 1)]))
            soft_targets = soft_targets.to(device)
            loss_task = Classifier.softmax_and_cross_entropy(Y_hat_logits,
                                                             soft_targets)
        else:
            loss_task = Classifier.logit_cross_entropy_loss(Y_hat_logits, T)

        # Compute gradients based on task loss (those might be used in the CL
        # regularizer).
        # print('confid.soft_targets', config.soft_targets)
        loss_task.backward(retain_graph=calc_reg, create_graph=calc_reg and config.backprop_dt)
        # print('loss_task',loss_task)

        # The current task embedding only depends in the task loss, so we can
        # update it already.
        if emb_optimizer is not None:
            emb_optimizer.step()

        #############################
        ### CL (HNET) Regularizer ###
        #############################
        loss_reg = 0
        dTheta = None

        if calc_reg:
            if config.no_lookahead:
                dTembs = None
                dTheta = None
            else:
                dTheta = opstep.calc_delta_theta(theta_optimizer, False,
                                                 lr=config.lr, detach_dt=not config.backprop_dt)

                if config.continue_emb_training:
                    dTembs = dTheta[-task_id:]
                    dTheta = dTheta[:-task_id]
                else:
                    dTembs = None

            '''
            #Use regularisation required to test the code
            loss_reg = hreg.calc_fix_target_reg(hnet, task_id,
                                                targets=targets_hypernet, dTheta=dTheta, dTembs=dTembs,
                                                mnet=mnet, inds_of_out_heads=regged_outputs,
                                                prev_theta=prev_theta, prev_task_embs=prev_task_embs,
                                                batch_size=config.cl_reg_batch_size, fisher_estimates=fisher)
            loss_reg = hreg.calc_fix_target_reg_online_ewc(hnet, task_id,
                                                targets=targets_hypernet, dTheta=dTheta, dTembs=dTembs,
                                                mnet=mnet, inds_of_out_heads=regged_outputs,
                                                prev_theta=prev_theta, prev_task_embs=prev_task_embs,
                                                batch_size=config.cl_reg_batch_size, fisher_estimates=fisher)
            
            loss_reg = ewc.ewc_regularizer(task_id, weights, None, hnet=hnet, online=True, gamma=1)
            
            loss_reg = hreg.calc_fix_target_reg(hnet, task_id,
                                                targets=targets_hypernet, dTheta=dTheta, dTembs=dTembs,
                                                mnet=mnet, inds_of_out_heads=regged_outputs,
                                                prev_theta=prev_theta, prev_task_embs=prev_task_embs,
                                                batch_size=config.cl_reg_batch_size, fisher_estimates=fisher)
            '''

            loss_reg = ewc.ewc_regularizer(task_id, theta_params, None, hnet=hnet, online=True, gamma=1)

            loss_reg *= config.beta

            loss_reg.backward()
            # total_loss=loss_task + loss_reg
            # total_loss.backward()
        # Now, that we computed the regularizer, we can use the accumulated
        # gradients and update the hnet (or mnet) parameters.
        theta_optimizer.step()
        Y_hat = F.softmax(Y_hat_logits, dim=1)
        classifier_accuracy = Classifier.accuracy(Y_hat, T) * 100.0

        #########################
        # Learning rate scheduler
        #########################
        if config.plateau_lr_scheduler:
            assert (iter_per_epoch != -1)
            if i % iter_per_epoch == 0 and i > 0:
                curr_epoch = i // iter_per_epoch
                logger.info('Computing test accuracy for plateau LR ' +
                            'scheduler (epoch %d).' % curr_epoch)
                # We need a validation quantity for the plateau LR scheduler.
                # FIXME we should use an actual validation set rather than the
                # test set.
                # Note, https://keras.io/examples/cifar10_resnet/ uses the test
                # set to compute the validation loss. We use the "validation"
                # accuracy instead.
                # FIXME We increase `train_iter` as the print messages in the
                # test method suggest that the testing has been executed before
                test_acc, _ = test(task_id, data, mnet, hnet, device, shared,
                                   config, writer, logger, theta_saved=None, train_iter=i + 1)
                mnet.train()
                if hnet is not None:
                    hnet.train()

                plateau_scheduler_theta.step(test_acc)
                if plateau_scheduler_emb is not None:
                    plateau_scheduler_emb.step(test_acc)

        if config.lambda_lr_scheduler:
            assert (iter_per_epoch != -1)
            if i % iter_per_epoch == 0 and i > 0:
                curr_epoch = i // iter_per_epoch
                logger.info('Applying Lambda LR scheduler (epoch %d).'
                            % curr_epoch)

                lambda_scheduler_theta.step()
                if lambda_scheduler_emb is not None:
                    lambda_scheduler_emb.step()

        ###########################
        ### Tensorboard summary ###
        ###########################
        # We don't wanna slow down training by having too much output.
        if i % 50 == 0:
            writer.add_scalar('train/task_%d/class_accuracy' % task_id,
                              classifier_accuracy, i)
            writer.add_scalar('train/task_%d/loss_task' % task_id, loss_task, i)
            writer.add_scalar('train/task_%d/loss_reg' % task_id, loss_reg, i)

        ### Show the current training progress to the user.
        if i % config.val_iter == 0:
            msg = 'Training step {}: Classifier Accuracy: {:.3f} ' + \
                  '(on current training batch).'
            logger.debug(msg.format(i, classifier_accuracy))

        iter_end_time = time()
        summed_iter_runtime += (iter_end_time - iter_start_time)

        if i % 200 == 0:
            logger.info('Training step: %d ... Done -- (runtime: %f sec)' % \
                        (i, iter_end_time - iter_start_time))

        ## Estimate diagonal Fisher elements.--------------------------------------------------
    # print('hnet.theta before', hnet.theta)
    if task_id != 5 and config.train_from_scratch is False and fisher is not None:
        fisher = ewc.fisher_mat(task_id, data, the, device, mnet, hnet=hnet,
                                empirical_fisher=True, online=True, gamma=1,
                                n_max=-1, regression=False,
                                allowed_outputs=None, logger=logger)
    '''
    fisher = ewc.compute_fisher(task_id, data, hnet.theta, device, mnet, hnet=hnet,
                            empirical_fisher=True, online=False, gamma=1,
                            n_max=-1, regression=False,
                            allowed_outputs=None, logger=logger)
     '''

    if mnet.batchnorm_layers is not None:
        if not config.bn_distill_stats and \
                not config.bn_no_running_stats and \
                not config.bn_no_stats_checkpointing:
            # Checkpoint the current running statistics (that have been
            # estimated while training the current task).
            for bn_layer in mnet.batchnorm_layers:
                assert (bn_layer.num_stats == task_id + 1)
                bn_layer.checkpoint_stats()

    avg_iter_time = summed_iter_runtime / config.n_iter
    logger.info('Average runtime per training iteration: %f sec.' % \
                avg_iter_time)

    logger.info('Elapsed time for training task %d: %f sec.' % \
                (task_id + 1, time() - start_time))
    return fisher, weights


def test_multiple(dhandlers, mnet, hnet, device, config, shared, writer,
                  logger, theta_saving):
    """Method to test continual learning experiment accuracy

    Args:
        (....): See docstring of function :func:`train`.
        dhandlers (list): List of data handlers. The accuracy of each task in
            this list will be computed using function :func:`test`. The index
            within the list will be considered as task ID.
    """
    class_accs = []
    task_accs = []

    num_tasks = len(dhandlers)

    ### Task-incremental learning
    if config.cl_scenario == 1:
        logger.info('### Testing task-incremental learning scenario')
        # Iterate through learned embeddings and tasks and compute test acc.
        for j in range(num_tasks):
            data = dhandlers[j]

            test_acc, _ = test(j, data, mnet, hnet, device, shared,
                               config, writer, logger, theta_saved=theta_saving)

            class_accs.append(test_acc)
            shared.summary['acc_final'][j] = test_acc

        shared.summary['acc_avg_final'] = np.mean(class_accs)
        logger.info('### Task-incremental learning scenario accuracies: %s ' \
                    % (str(class_accs)) + '(avg: %.3f)'
                    % (shared.summary['acc_avg_final']))

        writer.add_scalar('final/task_incremental',
                          shared.summary['acc_avg_final'])

    ### Domain-incremental learning & class-incremental learning
    if config.cl_scenario == 2 or config.cl_scenario == 3:
        raise NotImplementedError()

        if config.cl_scenario == 2:
            logger.info('### Testing domain-incremental learning scenario')
        else:
            logger.info('### Testing class-incrementa learning scenario')

        for j in range(num_tasks):
            data = dhandlers[j]

            test_acc, task_acc = test(j, data, mnet, hnet, device, shared,
                                      config, writer, logger, theta_saved=theta_saving)

            class_accs.append(test_acc)
            task_accs.append(task_acc)

            shared.summary['acc_final'][j] = test_acc

        shared.summary['acc_avg_final'] = np.mean(class_accs)

        if config.cl_scenario == 2:
            logger.info('### Domain-incremental learning scenario ' +
                        'accuracies: %s ' % (str(class_accs)) + '(avg: %.3f)'
                        % (shared.summary['acc_avg_final']))
            writer.add_scalar('final/domain_incremental',
                              shared.summary['acc_avg_final'])
        else:
            logger.info('### Class-incremental learning scenario ' +
                        'accuracies: %s ' % (str(class_accs)) + '(avg: %.3f)'
                        % (shared.summary['acc_avg_final']))
            writer.add_scalar('final/class_incremental',
                              shared.summary['acc_avg_final'])

        logger.info('### Task-inference accuracies: %s ' \
                    % (str(task_accs)) + '(avg: %.3f)'
                    % (np.mean(task_accs)))
        writer.add_scalar('final/task_inference_acc', np.mean(task_accs))

    return task_accs, class_accs


def analysis(dhandlers, mnet, hnet, device, config, shared, writer, logger,
             during_weights):
    """A function to do some post-hoc analysis on the hypernetwork.

    Specifically, this function does the following:
        - Computing and logging statistics on how the weights changed since a
          task has been learned.
        - Assessing the diversity of ``hnet`` outputs, i.e., how close are the
          ``hnet`` outputs for different tasks.

    Args:
        (....): See docstring of function :func:`test_multiple`.
        during_weights (list): List of flattened ``hnet`` outputs right after
            training on each task.
    """
    assert hnet is not None
    mnet.eval()
    hnet.eval()

    num_tasks = len(dhandlers)

    # Test how much the weights of each task have changed during training the
    # remaining tasks.
    for j in range(num_tasks):
        cur_weights = hnet.forward(j)
        cur_weights = torch.cat([a.detach().clone().cpu().flatten()
                                 for a in cur_weights])
        aft_weights = torch.cat([a.flatten() for a in during_weights[j]])

        logger.info('### Euclidean distance of current hnet output to ' +
                    'original one for task %d: %f' % \
                    (j, torch.sqrt(torch.sum((aft_weights - cur_weights) ** 2))))

    # FIXME Inefficient, we already computed all hnet outputs above.
    for j in range(num_tasks):
        for i in range(num_tasks):
            if i <= j:
                continue
            weights_1 = hnet.forward(j)
            weights_2 = hnet.forward(i)
            weights_1 = torch.cat([a.detach().clone().flatten() \
                                   for a in weights_1])
            weights_2 = torch.cat([a.detach().clone().flatten() \
                                   for a in weights_2])
            logger.info('### Euclidean distance between ' +
                        'task %d and task %d: %f' % (j, i,
                                                     torch.sqrt(torch.sum((weights_1 - weights_2) ** 2))))


def run(config, experiment='resnet'):
    """Run the training.

    Args:
        config (argparse.Namespace): Command-line arguments.
        experiment (str): Which kind of experiment should be performed?

            - ``resnet``: CIFAR-10/100 with Resnet-32.
            - ``zenke``: CIFAR-10/100 with Zenkenet.
    """
    assert (experiment in ['resnet', 'zenke'])

    script_start = time()

    device, writer, logger = sutils.setup_environment(config,
                                                      logger_name='det_cl_cifar_%s' % experiment)
    # TODO Adapt script to allow checkpointing of models using
    # `utils.torch_ckpts` (i.e., we should be able to continue training or just
    # test an existing checkpoint).
    # config.ckpt_dir = os.path.join(config.out_dir, 'checkpoints')

    # Container for variables shared across function.
    shared = Namespace()
    shared.experiment = experiment

    ### Load datasets (i.e., create tasks).
    dhandlers = tutils.load_datasets(config, shared, logger,
                                     data_dir='../datasets')
    print(
        '-----------------before tutils.get_main_model()------------------------------------------------------------------------------------------------------')
    ### Create main network.
    # TODO Allow main net only training.
    config.mnet_only = True
    mnet = tutils.get_main_model(config, shared, logger, device, no_weights=not config.mnet_only)

    print(
        '------------------------beforetutils.get_hnet_model()-------------------------------------------------------------------------------------------------')
    ### Create the hypernetwork.

    if config.mnet_only:
        hnet = None
    else:
        hnet = tutils.get_hnet_model(config, mnet, logger, device)
    print(
        '--------------------------------------------------------------------------------------------------------------------------')
    ### Initialize the performance measures, that should be tracked during
    ### training.
    tutils.setup_summary_dict(config, shared, mnet, hnet=hnet)

    # Add hparams to tensorboard, such that the identification of runs is
    # easier.
    writer.add_hparams(hparam_dict={**vars(config), **{
        'num_weights_main': shared.summary['num_weights_main'],
        'num_weights_hyper': shared.summary['num_weights_hyper'],
        'num_weights_ratio': shared.summary['num_weights_ratio'],
    }}, metric_dict={})

    # FIXME: Method "calc_fix_target_reg" expects a None value.
    # But `writer.add_hparams` can't deal with `None` values.
    if config.cl_reg_batch_size == -1:
        config.cl_reg_batch_size = None

    # We keep the hnet output right after training to measure forgetting.
    weights_after_training = []

    ######################
    ### Start Training ###
    ######################
    fisher = []
    prev_weights_fish = []
    theta_saving = []
    for j in range(config.num_tasks):
        logger.info('Starting training of task %d ...' % (j + 1))

        data = dhandlers[j]
        print('config.init_with_prev_emb', config.init_with_prev_emb)
        # It might be that tasks are very similar and we can transfer knowledge
        # form the previous solution.
        if hnet is not None and config.init_with_prev_emb and j > 0:
            last_emb = hnet.get_task_emb(j - 1).detach().clone()
            hnet.get_task_emb(j).data = last_emb

        # Training from scratch -- create new network instance!
        # -> No transfer possible.
        # config.train_from_scratch =
        print('config.train_from_scratch', config.train_from_scratch)

        if j > 0 and config.train_from_scratch:
            print('config.train_from_scratch', config.train_from_scratch)
            # FIXME Since we simply override the current network, future testing
            # on this new network for old tasks doesn't make sense. So we
            # shouldn't report `final` accuracies.
            if config.mnet_only:
                logger.info('From scratch training: Creating new main network.')
                mnet = tutils.get_main_model(config, shared, logger, device,
                                             no_weights=not config.mnet_only)
            else:
                logger.info('From scratch training: Creating new hypernetwork.')
                hnet = tutils.get_hnet_model(config, mnet, logger, device)

        ################################
        ### Train and test on task j ###
        ################################
        fish, prev_weights_fish = train(j, data, mnet, hnet, device, config, shared, writer, logger, fisher,
                                        prev_weights_fish)
        fisher.append(fish)
        # print('fisher matrix', fisher)
        print('length of fisher matrix', len(fisher))
        params_to_regularize = list(hnet.theta)
        for d in params_to_regularize:
            print(d.shape)
        # saving entire theta but still the hidden paramters frozen in LSTM(LSTM_cell.py). More memory efficient code is to save only
        theta_saving.append([d.detach().clone() for d in params_to_regularize])
        # print('theta_saved', theta_saving[0])
        ### Final test run.
        if hnet is not None:
            weights = hnet.forward(j)
            # Push to CPU to avoid growing GPU memory when solving very long
            # task sequences.
            weights = [w.detach().clone().cpu() for w in weights]
            weights_after_training.append(weights)

        test_acc, _ = test(j, data, mnet, hnet, device, shared, config, writer, logger, theta_saved=theta_saving)
        logger.info('### Accuracy of task %d / %d:  %.3f' % \
                    (j + 1, config.num_tasks, test_acc))
        logger.info('### Finished training task: %d' % (j + 1))
        shared.summary['acc_during'][j] = test_acc

        # Backup results so far.
        tutils.save_summary_dict(config, shared, experiment)

    shared.summary['acc_avg_during'] = np.mean(shared.summary['acc_during'])

    logger.info('### Accuracy of individual tasks after training %s' % \
                (str(shared.summary['acc_during'])))
    logger.info('### Average of these accuracies  %.2f' % \
                (shared.summary['acc_avg_during']))
    writer.add_scalar('final/during_acc_avg', shared.summary['acc_avg_during'])

    #########################################
    ### Test continual learning scenarios ###
    #########################################
    test_multiple(dhandlers, mnet, hnet, device, config, shared, writer,
                  logger, theta_saving)

    #########################
    ### Run some analysis ###
    #########################
    if not config.mnet_only:
        analysis(dhandlers, mnet, hnet, device, config, shared, writer, logger,
                 weights_after_training)

    ### Write final summary.
    shared.summary['finished'] = 1
    tutils.save_summary_dict(config, shared, experiment)

    writer.close()

    logger.info('Program finished successfully in %f sec.'
                % (time() - script_start))


if __name__ == '__main__':
    raise Exception('Script is not executable!')
