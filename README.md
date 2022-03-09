# HJI_VI


The optimization uses casadi as an interface and ipopt as solver. 


`d_star_solution` solves for optimum adversary control

`u_star_solution` solves for optimum ego control 


## Generating Trajectory

To generate the trajectory using the value network, first download the checkpoints as mentioned in the [deepreach](/deepreach) submodule. Then run the generate_trajectory with appropriate path settings (line #22 and #238). 
