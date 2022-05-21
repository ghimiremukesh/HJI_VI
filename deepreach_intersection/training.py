'''Implements a generic training loop.
'''

import torch
import utils
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm
import time
import numpy as np
import os
import shutil


# modify to train two networks

def train(model_a, model_na, train_dataloader, epochs, lr, steps_til_summary, epochs_til_checkpoint, model_dir_a,
          model_dir_na, loss_fn_a, loss_fn_na,
          summary_fn=None, val_dataloader=None, double_precision=False, clip_grad=False, use_lbfgs=False,
          loss_schedules=None,
          validation_fn=None, start_epoch=0):
    optim_a = torch.optim.Adam(lr=lr, params=model_a.parameters())  # optimizer for aggressive model
    optim_na = torch.optim.Adam(lr=lr, params=model_na.parameters())  # optimizer for non-aggressive model

    # copy settings from Raissi et al. (2019) and here 
    # https://github.com/maziarraissi/PINNs
    if use_lbfgs:
        optim_a = torch.optim.LBFGS(lr=lr, params=model_a.parameters(), max_iter=50000, max_eval=50000,
                                    history_size=50, line_search_fn='strong_wolfe')
        optim_na = torch.optim.LBFGS(lr=lr, params=model_na.parameters(), max_iter=50000, max_eval=50000,
                                     history_size=50, line_search_fn='strong_wolfe')

    # Load the checkpoint if required
    if start_epoch > 0:
        # Load the model and start training from that point onwards
        # Train aggressive model_a first and then model_na

        # model_path = os.path.join(model_dir, 'checkpoints', 'model_epoch_%04d.pth' % start_epoch)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path_a = os.path.join(current_dir, 'validation_scripts', 'a', 'model_a_epoch_109000_20k.pth')
        model_path_na = os.path.join(current_dir, 'validation_scripts', 'na', 'model_na_epoch_109000_20k.pth')
        checkpoint_a = torch.load(model_path_a)
        checkpoint_na = torch.load(model_path_na)
        model_a.load_state_dict(checkpoint_a['model'])
        model_a.train()
        optim_a.load_state_dict(checkpoint_a['optimizer'])
        optim_a.param_groups[0]['lr'] = lr
        model_na.load_state_dict(checkpoint_na['model'])
        model_na.train()
        optim_na.load_state_dict(checkpoint_na['optimizer'])
        optim_na.param_groups[0]['lr'] = lr
        assert (start_epoch == checkpoint_a['epoch'] and start_epoch == checkpoint_na)
    else:
        # Start training from scratch
        if os.path.exists(model_dir_a) or os.path.exists(model_dir_na):
            val = input("The model directories %s, %s exist. Overwrite? (y/n)" % (model_dir_a, model_dir_na))
            if val == 'y':
                shutil.rmtree(model_dir_a)
                shutil.rmtree(model_dir_na)
        os.makedirs(model_dir_a)
        os.makedirs(model_dir_na)

    # TRAIN AGGRESSIVE FIRST ---> model_a
    summaries_dir_a = os.path.join(model_dir_a, 'summaries')
    summaries_dir_na = os.path.join(model_dir_na, 'summaries')
    utils.cond_mkdir(summaries_dir_a)
    utils.cond_mkdir(summaries_dir_na)

    checkpoints_dir_a = os.path.join(model_dir_a, 'checkpoints')
    checkpoints_dir_na = os.path.join(model_dir_na, 'checkpoints')
    utils.cond_mkdir(checkpoints_dir_a)
    utils.cond_mkdir(checkpoints_dir_na)

    writer_a = SummaryWriter(summaries_dir_a)
    writer_na = SummaryWriter(summaries_dir_na)

    total_steps = 0
    with tqdm(total=len(train_dataloader) * epochs - start_epoch) as pbar:
        train_losses_a = []
        train_losses_na = []
        for epoch in range(start_epoch, epochs):
            if not epoch % epochs_til_checkpoint and epoch:
                # Saving the optimizer state is important to produce consistent results
                checkpoint_a = {
                    'epoch': epoch,
                    'model': model_a.state_dict(),
                    'optimizer': optim_a.state_dict()}
                checkpoint_na = {
                    'epoch': epoch,
                    'model': model_na.state_dict(),
                    'optimizer': optim_na.state_dict()}

                torch.save(checkpoint_a,
                           os.path.join(checkpoints_dir_a, 'model_epoch_%04d.pth' % epoch))
                torch.save(checkpoint_na,
                           os.path.join(checkpoints_dir_na, 'model_epoch_%04d.pth' % epoch))
                # torch.save(model.state_dict(),
                #            os.path.join(checkpoints_dir, 'model_epoch_%04d.pth' % epoch))
                np.savetxt(os.path.join(checkpoints_dir_a, 'train_losses_epoch_%04d.txt' % epoch),
                           np.array(train_losses_a))
                np.savetxt(os.path.join(checkpoints_dir_na, 'train_losses_epoch_%04d.txt' % epoch),
                           np.array(train_losses_na))

                if validation_fn is not None:
                    validation_fn(model_a, checkpoints_dir_a, epoch)
                    validation_fn(model_na, checkpoints_dir_na, epoch)

            for step, (model_input, gt) in enumerate(train_dataloader):
                start_time = time.time()

                model_input = {key: value.cuda() for key, value in model_input.items()}
                gt = {key: value.cuda() for key, value in gt.items()}

                if double_precision:
                    model_input = {key: value.double() for key, value in model_input.items()}
                    gt = {key: value.double() for key, value in gt.items()}

                if use_lbfgs:
                    def closure_a():
                        # train aggressive
                        optim_a.zero_grad()
                        model_output_a = model_a(model_input)
                        losses_a = loss_fn_a(model_output_a, model_output_na, gt)
                        train_loss_a = 0.
                        for loss_name, loss in losses_a.items():
                            train_loss_a += loss.mean()
                        train_loss_a.backward()
                        return train_loss_a

                    def closure_na():
                        # train non-aggressive
                        optim_na.zero_grad()
                        model_output_na = model_a(model_input)
                        losses_na = loss_fn_na(model_output_a, model_output_na, gt)
                        train_loss_na = 0.
                        for loss_name, loss in losses_na.items():
                            train_loss_na += loss.mean()
                        train_loss_na.backward()
                        return train_loss_na

                    optim_a.step(closure_a)
                    optim_na.step(closure_na)

                model_output_a = model_a(model_input)
                model_output_na = model_na(model_input)

                losses_a = loss_fn_a(model_output_a, model_output_na, gt)
                losses_na = loss_fn_na(model_output_a, model_output_na, gt)

                # import ipdb; ipdb.set_trace()

                train_loss_a = 0.
                train_loss_na = 0.
                for loss_name, loss in losses_a.items():
                    single_loss = loss.mean()

                    if loss_schedules is not None and loss_name in loss_schedules:
                        writer_a.add_scalar(loss_name + "_weight", loss_schedules[loss_name](total_steps), total_steps)
                        single_loss *= loss_schedules[loss_name](total_steps)

                    writer_a.add_scalar(loss_name, single_loss, total_steps)
                    train_loss_a += single_loss

                for loss_name, loss in losses_na.items():
                    single_loss = loss.mean()

                    if loss_schedules is not None and loss_name in loss_schedules:
                        writer_na.add_scalar(loss_name + "_weight", loss_schedules[loss_name](total_steps), total_steps)
                        single_loss *= loss_schedules[loss_name](total_steps)

                    writer_na.add_scalar(loss_name, single_loss, total_steps)
                    train_loss_na += single_loss

                train_losses_a.append(train_loss_a.item())
                writer_a.add_scalar("total_train_loss", train_loss_a, total_steps)

                train_losses_na.append(train_loss_na.item())
                writer_na.add_scalar("total_train_loss", train_loss_na, total_steps)

                if not total_steps % steps_til_summary:
                    torch.save(model_a.state_dict(),
                               os.path.join(checkpoints_dir_a, 'model_current.pth'))
                    torch.save(model_na.state_dict(),
                               os.path.join(checkpoints_dir_na, 'model_current.pth'))
                    # summary_fn(model, model_input, gt, model_output, writer_a, total_steps)

                if not use_lbfgs:
                    optim_a.zero_grad()
                    train_loss_a.backward(retain_graph=True)
                    optim_na.zero_grad()
                    train_loss_na.backward(retain_graph=True)

                    if clip_grad:
                        if isinstance(clip_grad, bool):
                            torch.nn.utils.clip_grad_norm_(model_a.parameters(), max_norm=1.)
                            torch.nn.utils.clip_grad_norm_(model_na.parameters(), max_norm=1.)
                        else:
                            torch.nn.utils.clip_grad_norm_(model_a.parameters(), max_norm=clip_grad)
                            torch.nn.utils.clip_grad_norm_(model_na.parameters(), max_norm=clip_grad)

                    optim_a.step()
                    optim_na.step()

                pbar.update(1)

                if not total_steps % steps_til_summary:
                    tqdm.write("Epoch %d, Total loss Aggressive %0.6f, Total loss Non-Agg. %0.6f, iteration time %0.6f" % (
                        epoch, train_loss_a, train_loss_na, time.time() - start_time))

                    # NO validation data for this case. skipping update.....
                    if val_dataloader is not None:
                        print("Running validation set...")
                        model_a.eval()
                        with torch.no_grad():
                            val_losses = []
                            for (model_input, gt) in val_dataloader:
                                model_output = model_a(model_input)
                                val_loss = loss_fn_a(model_output, gt)
                                val_losses.append(val_loss)

                            writer_a.add_scalar("val_loss", np.mean(val_losses), total_steps)
                        model_a.train()

                total_steps += 1

        torch.save(model_a.state_dict(),
                   os.path.join(checkpoints_dir_a, 'model_final.pth'))
        torch.save(model_na.state_dict(),
                   os.path.join(checkpoints_dir_na, 'model_final.pth'))
        np.savetxt(os.path.join(checkpoints_dir_a, 'train_losses_final.txt'),
                   np.array(train_losses_a))
        np.savetxt(os.path.join(checkpoints_dir_na, 'train_losses_final.txt'),
                   np.array(train_losses_na))



class LinearDecaySchedule():
    def __init__(self, start_val, final_val, num_steps):
        self.start_val = start_val
        self.final_val = final_val
        self.num_steps = num_steps

    def __call__(self, iter):
        return self.start_val + (self.final_val - self.start_val) * min(iter / self.num_steps, 1.)
