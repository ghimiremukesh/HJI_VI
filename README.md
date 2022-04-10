# HJI_VI


The optimization uses casadi as an interface and ipopt as solver. 


`d_star_solution` solves for optimum adversary control

`u_star_solution` solves for optimum ego control 


## Generating Trajectory

To generate the trajectory using the value network, first download the checkpoints as mentioned in the [deepreach](https://github.com/smlbansal/deepreach/tree/b0666c1113c5bf235284ba9634781da92d2f3fab) submodule from their [google drive](https://drive.google.com/file/d/18VkOTctkzuYuyK2GRwQ4wmN92WhdXtvS/view?usp=sharing). Then run the generate_trajectory with appropriate path settings (line #22 and #238). 

## deepreach_intersection
This folder contains the code to train intersection case through deepreach approach

Set up a conda environment with all dependencies like so:

`conda env create -f environment.yml`

`conda activate siren`

### function of python file

`dataio.py`: generate train data for value network. This file includes two data types: HJI VI(BRAT) and HJI

`loss_functions.py`: loss function definiton for value network. This file includes two loss_function types: HJI VI(BRAT) and HJI

`modules.py`: network architecture setting

`training.py`: network training

`diff_operators.py`: include hessian, jacobian, etc. calculation

`training curve_plot.py`: plot training curve of network

`validation_scripts\closedloop_traj_generation.py`: use value network as controller to generate closed-loop trajectory

`validation_scripts\value_generation.py`: use value network and state from ground truth data to generate value and costate

`validation_scripts\value_landscape_generation.py`: use value network to generate value landscape

`validation_scripts\value_landscape_plot.py`: plot value landscape in the initial time

`validation_scripts\trajectory.py`: plot d1-d2 trajectory for ground truth data (or data generated by network)

`validation_scripts\generate_trajectory.py`: use BRT value network to generate closed-loop trajectory

`experiment_scripts\train_intersection_BRAT.py`: set up value network of BRAT training. If need pretrain (fit for boundary condition), the parameter can set as True
(default setting is True)  

`experiment_scripts\train_intersection_HJI.py`: set up value network of HJI training. If need pretrain (fit for boundary condition), the parameter can set as True (default setting is True)

### experiment running
* run `experiment_scripts\train_intersection_HJI.py` or `experiment_scripts\train_intersection_BRAT.py` to train value network. Data generation and loss definition
of HJI VI(BRAT) and HJI can review `dataio.py` and `loss_functions.py`. `modules.py`, `training.py` and `diff_operators.py` is based on siren paper without any change. 

* trained model and loss information is stored in `experiment_scripts\log\xxx\checkpoints`. Completed model is named `model_final.pth` and final train loss is named `train_losses_final.txt`. Run `training curve_plot.py` to check the train loss if need. Please make sure to put `train_losses_final.txt` in the correct folder.
 
* run `validation_scripts\value_landscape_plot.py` to check value landscape in d1-d2 figure. `validation_scripts` folder includes file to generate trajectory and value. Please review descriptions in `function of python file`.
