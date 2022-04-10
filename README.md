# HJI_VI


The optimization uses casadi as an interface and ipopt as solver. 


`d_star_solution` solves for optimum adversary control

`u_star_solution` solves for optimum ego control 


## Generating Trajectory

To generate the trajectory using the value network, first download the checkpoints as mentioned in the [deepreach](https://github.com/smlbansal/deepreach/tree/b0666c1113c5bf235284ba9634781da92d2f3fab) submodule from their [google drive](https://drive.google.com/file/d/18VkOTctkzuYuyK2GRwQ4wmN92WhdXtvS/view?usp=sharing). Then run the generate_trajectory with appropriate path settings (line #22 and #238). 

## Deepreach_intersection

`dataio.py`: generate train data for value network. This file includes two data types: HJI VI(BRAT) and HJI
`loss_functions.py`: loss function definiton for value network. This file includes two loss_function types: HJI VI(BRAT) and HJI
`validation_scripts\closedloop_traj_generation.py`: use value network as controller to generate closed-loop trajectory
`validation_scripts\value_generation.py`: use value network and state from ground truth to generate value and costate
`validation_scripts\value_landscape_generation.py`: use value network to generate value landscape
`validation_scripts\value_landscape_plot`: plot value landscape
