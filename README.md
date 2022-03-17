# HJI_VI


The optimization uses casadi as an interface and ipopt as solver. 


`d_star_solution` solves for optimum adversary control

`u_star_solution` solves for optimum ego control 


## Generating Trajectory

To generate the trajectory using the value network, first download the checkpoints as mentioned in the [deepreach](https://github.com/smlbansal/deepreach/tree/b0666c1113c5bf235284ba9634781da92d2f3fab) submodule from their [google drive](https://drive.google.com/file/d/18VkOTctkzuYuyK2GRwQ4wmN92WhdXtvS/view?usp=sharing). Then run the generate_trajectory with appropriate path settings (line #22 and #238). 
