# Enable import from parent package
import matplotlib
import numpy as np

matplotlib.use('Agg')

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dataio, utils, training, loss_functions, modules, training_selfsupervised

from torch.utils.data import DataLoader
import configargparse
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

p.add_argument('--logging_root', type=str, default='./logs', help='root for logging')
p.add_argument('--experiment_name', type=str, required=False,
               help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')

# General training options
p.add_argument('--batch_size', type=int, default=32)
p.add_argument('--lr', type=float, default=2e-5, help='learning rate. default=2e-5')

# 200000 for hybrid, 100000 for supervised, 160000 for self-supervised
p.add_argument('--num_epochs', type=int, default=10000,
               help='Number of epochs to train for.')

p.add_argument('--epochs_til_ckpt', type=int, default=1000,
               help='Time interval in seconds until checkpoint is saved.')
p.add_argument('--steps_til_summary', type=int, default=100,
               help='Time interval in seconds until tensorboard summary is saved.')
p.add_argument('--model', type=str, default='tanh', required=False, choices=['sine', 'tanh', 'sigmoid', 'relu'],
               help='Type of model to evaluate, default is sine.')
p.add_argument('--mode', type=str, default='mlp', required=False, choices=['mlp', 'rbf', 'pinn'],
               help='Whether to use uniform velocity parameter')
p.add_argument('--tMin', type=float, default=0.0, required=False, help='Start time of the simulation')
p.add_argument('--tMax', type=float, default=3, required=False, help='End time of the simulation')
p.add_argument('--num_hl', type=int, default=3, required=False, help='The number of hidden layers')
p.add_argument('--num_nl', type=int, default=64, required=False, help='Number of neurons per hidden layer.')
p.add_argument('--pretrain_iters', type=int, default=1000, required=False, help='Number of pretrain iterations')
p.add_argument('--counter_start', type=int, default=-1, required=False,
               help='Defines the initial time for the curriculul training')
p.add_argument('--counter_end', type=int, default=9000, required=False,
               help='Defines the linear step for curriculum training starting from the initial time')
p.add_argument('--num_src_samples', type=int, default=10000, required=False,
               help='Number of source samples at each time step')

p.add_argument('--collisionR', type=float, default=0.25, required=False, help='Collision radius between vehicles')
p.add_argument('--minWith', type=str, default='target', required=False, choices=['none', 'zero', 'target'],
               help='BRS vs BRT computation')

p.add_argument('--clip_grad', default=0.0, type=float, help='Clip gradient.')
p.add_argument('--use_lbfgs', default=False, type=bool, help='use L-BFGS.')
p.add_argument('--pretrain', action='store_true', default=True, required=False, help='Pretrain dirichlet conditions')

p.add_argument('--seed', type=int, default=0, required=False, help='Seed for the simulation.')

p.add_argument('--checkpoint_path', default=None, help='Checkpoint to trained model.')
p.add_argument('--checkpoint_toload', type=int, default=0, help='Checkpoint from which to restart the training.')
opt = p.parse_args()

# Set the source coordinates for the target set and the obstacle sets

source_coords = [0., 0., 0., 0.]
if opt.counter_start == -1:
    opt.counter_start = opt.checkpoint_toload

if opt.counter_end == -1:
    opt.counter_end = opt.num_epochs

'''
Using HJI to train the value network
'''
dataset = dataio.IntersectionHJI_SelfSupervised(numpoints=61000,
                                                pretrain=opt.pretrain, tMin=opt.tMin,
                                                tMax=opt.tMax, counter_start=opt.counter_start,
                                                counter_end=opt.counter_end,
                                                pretrain_iters=opt.pretrain_iters, seed=opt.seed,
                                                num_src_samples=opt.num_src_samples)

dataloader = DataLoader(dataset, shuffle=True, batch_size=opt.batch_size, pin_memory=True, num_workers=0)

model = modules.SingleBVPNet(in_features=5, out_features=1, type=opt.model, mode=opt.mode,
                             final_layer_factor=1., hidden_features=opt.num_nl, num_hidden_layers=opt.num_hl)

model.to(device)

# Define the loss
'''
loss definition for HJI
'''
# root_path = os.path.join(opt.logging_root, 'experiment_grow_gamma_tanh_try/')

# for loop over the training iterations for each gamma

# gamma = np.linspace(0.001, 5, 50)
gamma = np.logspace(0.001, 5, 10, base=10)/20000
# epoch_each = (opt.counter_end / len(gamma)) + opt.pretrain_iters

for i in range(len(gamma)):
    if i == 0:
        load_dir = None
        start_epoch = 0
    else:
        load_dir = os.path.join(opt.logging_root, f'logspace-small/experiment_grow_gamma_tanh_try_gamma_{gamma[i - 1]}/')
        start_epoch = opt.num_epochs-1

    root_path = os.path.join(opt.logging_root, f'logspace-small/experiment_grow_gamma_tanh_try_gamma_{gamma[i]}/')

    loss_fn = loss_functions.initialize_intersection_HJI_selfsupervised(dataset, gamma[i])

    print(f'\nTraining with gamma: {gamma[i]}\n')

    training_selfsupervised.train(model=model, train_dataloader=dataloader, epochs=opt.num_epochs, lr=opt.lr,
                                  steps_til_summary=opt.steps_til_summary, epochs_til_checkpoint=opt.epochs_til_ckpt,
                                  model_dir=root_path, loss_fn=loss_fn, clip_grad=opt.clip_grad,
                                  use_lbfgs=opt.use_lbfgs, validation_fn=None, load_dir=load_dir,
                                  start_epoch=start_epoch)