import collections
import datetime
import logging
import os
import time
import torch
import torch.distributed as dist

from ssd.engine.inference import do_evaluation
from ssd.utils import dist_util
from ssd.utils.metric_logger import MetricLogger

import numpy as np
from tqdm import tqdm

def write_metric(eval_result, prefix, summary_writer, global_step):
    for key in eval_result:
        value = eval_result[key]
        tag = '{}/{}'.format(prefix, key)
        if isinstance(value, collections.Mapping):
            write_metric(value, tag, summary_writer, global_step)
        else:
            summary_writer.add_scalar(tag, value, global_step=global_step)


def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = dist_util.get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses

def do_train(cfg, model,
             data_loader,
             optimizer,
             scheduler,
             checkpointer,
             device,
             arguments,
             args):
    logger = logging.getLogger("SSD.trainer")
    logger.info("Start training ...")
    meters = MetricLogger()

    model.train()
    save_to_disk = dist_util.get_rank() == 0
    if args.use_tensorboard and save_to_disk:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            from tensorboardX import SummaryWriter
        summary_writer = SummaryWriter(log_dir=os.path.join(cfg.OUTPUT_DIR, 'tf_logs'))
    else:
        summary_writer = None

    if cfg.withda:
        max_iter = len(data_loader[0])
        iter2 = iter(data_loader[1])
        data_loader_ = data_loader[0]


    else:
        max_iter = len(data_loader)
        data_loader_ = data_loader

    start_iter = arguments["iteration"]
    start_training_time = time.time()
    end = time.time()

    for iteration, (images, targets, _) in enumerate(data_loader_, start_iter):
        iteration = iteration + 1
        arguments["iteration"] = iteration


        if cfg.withda and iteration/max_iter> 0:

            try:
                images2, targets2, _ = next(iter2)
            except StopIteration:
                iter2 = iter(data_loader[1])
                images2, targets2, _  = next(iter2)

            images2 = images2.to(device)

        images = images.to(device)
        targets = targets.to(device)

        if cfg.withda and iteration/max_iter> 0 :
            p = iteration/ max_iter
            if cfg.DA.CONST == -1:
                constantg = torch.tensor(2. / (1. + np.exp(-10 * p)) - 1).to(images.device)
                constantl = torch.tensor(2. / (1. + np.exp(-3 * p)) - 1).to(images.device)

                if p>=0.8:
                    constantg = torch.tensor(2. / (1. + np.exp(3 * (p - 0.8)))).to(images.device)

                constant = [constantg,constantl]
            else:
                if len(cfg.DA.FIXED_CONST) == 1:
                    if cfg.DA.FIXED_CONST[0] == -1:
                        constant = torch.tensor(2. / (1. + np.exp(-cfg.DA.CONST * p)) - 1).to(images.device)
                    else:
                        constant = cfg.DA.FIXED_CONST[0]
                else:
                    if p<0.1:
                        constant = cfg.DA.FIXED_CONST[0]
                    else:
                        constant = cfg.DA.FIXED_CONST[1]

                    constant = [constant,iteration,max_iter]


            loss_dict1 = model(images, targets=targets, constant =constant, domain='s')

            loss_dict2 = model(images2, targets=None, constant=constant, domain='t')

            for k in loss_dict2.keys():
                    loss_dict1[k+'2'] = loss_dict2[k]



            loss_dict = loss_dict1

        else:

            loss_dict = model(images, targets=targets)


        loss = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss.item() for loss in loss_dict_reduced.values())
        meters.update(total_loss=losses_reduced, **loss_dict_reduced)

        optimizer.zero_grad()
        try:
            loss.backward()
        except Exception as e:
            print(e)
            #import pdb;pdb.set_trace()

        optimizer.step()
        if cfg.SOLVER.TYPE != 'adam':
            scheduler.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time)
        if iteration % args.log_step == 0:
            eta_seconds = meters.time.global_avg * (max_iter - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            logger.info(
                meters.delimiter.join([
                    "iter: {iter:06d}",
                    "lr: {lr:.5f}",
                    '{meters}',
                    "eta: {eta}",
                    'mem: {mem}M',
                ]).format(
                    iter=iteration,
                    lr=optimizer.param_groups[0]['lr'],
                    meters=str(meters),
                    eta=eta_string,
                    mem=round(torch.cuda.max_memory_allocated() / 1024.0 / 1024.0),
                )
            )
            if summary_writer:
                global_step = iteration
                summary_writer.add_scalar('losses/total_loss', losses_reduced, global_step=global_step)
                for loss_name, loss_item in loss_dict_reduced.items():
                    summary_writer.add_scalar('losses/{}'.format(loss_name), loss_item, global_step=global_step)
                summary_writer.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=global_step)

        if iteration % args.save_step == 0:
            checkpointer.save("model_{:06d}".format(iteration), **arguments)

        if args.eval_step > 0 and (iteration) % args.eval_step == 0 and not iteration == max_iter:
        # if True:
            with torch.no_grad():
                eval_results = do_evaluation(cfg, model, distributed=args.distributed, iteration=iteration)
                if dist_util.get_rank() == 0 and summary_writer:
                    for eval_result, dataset in zip(eval_results, cfg.DATASETS.TEST):
                        write_metric(eval_result['metrics'], 'metrics/' + dataset, summary_writer, iteration)
            model.train()  # *IMPORTANT*: change to train mode after eval.

    checkpointer.save("model_final", **arguments)
    # compute training time
    total_training_time = int(time.time() - start_training_time)
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info("Total training time: {} ({:.4f} s / it)".format(total_time_str, total_training_time / max_iter))
    return model
