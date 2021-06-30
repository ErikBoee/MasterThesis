# Master oppgave
The source directory is sorted into three directories and four additional files:

Directories:

create_experiments:

contains files to create experiments, initializer_svg.py needs
to have two svg paths to specify initial guess and true solution.
create_experiment.py creates several problems to compare, create_problem_dictionary.py
creates a single problem to run. 

display_experiments:

contains files to plot reconstructions and solution, and also to
calculate a metric for all experiments.
All performed experiments are currently moved to test.

run_experiments:

Files designed to run experiments on NTNUs idun server.
This way all experiments can be ran simultaneously.

The remaining four files are optimization_object_bfgs.py, 
and optimization_object_bfgs_utilities.py which contains most of the code and the method
for solving the problem, and constants.py and parameters.py which contains fixed constants
and parameters.

