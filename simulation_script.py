'''ACCESS Example User Simulation Script

Must define one simulation whose parameters will be optimised this way:

    1. Use a variable called "parameters" to define this simulation's free /
       optimisable parameters. Create it using `coexist.create_parameters`.
       You can set the initial guess here.

    2. The `parameters` creation should be fully self-contained between
       `#### ACCESS PARAMETERS START` and `#### ACCESS PARAMETERS END`
       blocks (i.e. it should not depend on code ran before that).

    3. By the end of the simulation script, define a variable called `error` -
       one number representing this simulation's error value.

Importantly, use `parameters.at[<free parameter name>, "value"]` to get this
simulation's free / optimisable variable values.
'''

# Either run the actual GranuDrum simulation (takes ~40 minutes) or extract
# pre-computed example data and instantly show error value and plots
run_simulation = True       # Run simulation (True) or use example data (False)
save_data = False           # Save particle positions, radii and timestamps
show_plots = False          # Show plots of simulated & experimental GranuDrum


#### ACCESS PARAMETERS START
from typing import Tuple

import os
import numpy as np
import cv2
import pandas as pd

import coexist
import konigcell as kc      # For occupancy grid computation

print(os.getcwd())
os.system("cp -r src access_seed1002") 
from src import main_run
print("Imported src successfully")


parameters = coexist.create_parameters(
    variables = ["cor","sf","ce_pp","ce_pw","fill"],
    minimums = [0.05, 0.05, 0, 0, 0.8],
    maximums = [1.0, 1.0, 500000, 500000, 1.20],
    values =   [0.5, 0.5, 250000, 250000, 1.0]			# fill should go to 1.027 as that was the measured.
)

access_id = 0               # Unique ID for each ACCESS simulation run
#### ACCESS PARAMETERS END

#res_file = open(sys.argv[2], "x")

# Extract current free parameters' values
cor = parameters.at["cor", "value"]
sf = parameters.at["sf", "value"]
ce_pp = parameters.at["ce_pp", "value"]
ce_pw = parameters.at["ce_pw", "value"]
fill = parameters.at["fill", "value"]

def simulate_mill(cor,sf,ce_pp,ce_pw,fill):
    ######## ~~ Need to work out how to change

    # Create a new LIGGGHTS simulation script with the parameter values above; read
    # in the simulation template and change the relevant lines
    with open("cohesion_template.liggghts", "r") as f:
        sim_script = f.readlines()

    sim_script[8] = f"log simulation_inputs/vsm_run_{access_id:0>6}.log\n"

    sim_script[47] = f"variable CoR11 equal {cor}\n"
    sim_script[48] = f"variable CoR12 equal {cor}\n"
    sim_script[49] = f"variable CoR21 equal {cor}\n"

    sim_script[52] = f"variable sf11 equal {sf}\n"
    sim_script[53] = f"variable sf12 equal {sf}\n"
    sim_script[54] = f"variable sf21 equal {sf}\n"

    sim_script[62] = f"variable ce11 equal {ce_pp}\n"
    sim_script[63] = f"variable ce12 equal {ce_pw}\n"
    sim_script[64] = f"variable ce21 equal {ce_pw}\n"

    sim_script[83] = f"variable fillmass equal {fill}\n"

    sim_script[149] = f"shell mkdir Outputs/Output_{access_id:0>6}\n"

    sim_script[174] = f"dump dmp all custom/vtk 20000 Outputs/Output_{access_id:0>6}/particles_*.vtk id type x y z ix iy iz vx vy vz fx fy fz omegax omegay omegaz radius\n"

    # Save the simulation template with the changed free parameters
    filepath = f"/rds/projects/w/windowcr-mondelez-milling/stirred_milling/ACCES_100cst/Inputs/vsm_{access_id:0>6}.sim"
    with open(filepath, "w") as f:
        f.writelines(sim_script)

    #sim = coexist.LiggghtsSimulation(filepath,parameters)
    slurm_command = f"mpirun -np 4 liggghts -in /rds/projects/w/windowcr-mondelez-milling/stirred_milling/ACCES_100cst/Inputs/vsm_{access_id:0>6}.sim"
    os.system(slurm_command)

    ##################################################################################

    # Make the analysis folder
    analysis_dir = f"Analysis/Analysis_{access_id:0>6}"
    if not os.path.exists(analysis_dir):
        os.mkdir(analysis_dir)

    # Creating the analysis data to do the pept velocities
    with open("run_cohesion_template.py", "r") as f:
        sim_script = f.readlines()

    sim_script[21] = f"input_filename = \"run_cohesion_{access_id:0>6}\"\n"
    sim_script[28] = f"output_folder = \"Outputs/Output_{access_id:0>6}\"\n"
    sim_script[29] = f"analysis_folder = \"Analysis/Analysis_{access_id:0>6}\"\n"

    filepath = f"access_seed1002/run_cohesion_{access_id:0>6}.py"
    with open(filepath, "w") as f:
        f.writelines(sim_script)

    ###################################################################################

    # Run the analysis script

    with open(f"access_seed1002/run_cohesion_{access_id:0>6}.py") as f:
        exec(f.read())

    os.chdir("/rds/projects/w/windowcr-mondelez-milling/stirred_milling/ACCES_100cst/")

    ###################################################################################

    # Using the analysis data to find the error
    pept_vel_av = pd.read_csv('/rds/projects/w/windowcr-mondelez-milling/stirred_milling/ACCES_100cst/ideal_pept100_400.csv')

    dem_vel_filename = f"/rds/projects/w/windowcr-mondelez-milling/stirred_milling/ACCES_100cst/Analysis/Analysis_{access_id:0>6}/cart_voxel_velocity_time_0.csv"
    dem_vel = pd.read_csv(dem_vel_filename)

    dem_num_filename = f"/rds/projects/w/windowcr-mondelez-milling/stirred_milling/ACCES_100cst/Analysis/Analysis_{access_id:0>6}/cart_voxel_number_time_0.csv"
    dem_num = pd.read_csv(dem_num_filename)

    X=12; Y=24; Z=36

    rolling_checker_Z = 0
    rolling_checker_Y = 0
    rolling_checker_X = 0

    velocity_list_dem = np.zeros(int(X*Z))
    freq_list_dem = np.zeros(int(X*Z))

    rolling_checker_Z = -1

    for i in range(int(X*Y*Z)):
        if rolling_checker_Z == Z-1:
            rolling_checker_Z = -1
            if rolling_checker_Y == Y-1:
                rolling_checker_Y = 0
                rolling_checker_X = rolling_checker_X + 1
            else:
                rolling_checker_Y = rolling_checker_Y + 1

        rolling_checker_Z = rolling_checker_Z + 1
        bin_number_xz = Z*rolling_checker_X + rolling_checker_Z

        freq_list_dem[bin_number_xz] += dem_num.Frequency[i]
        velocity_list_dem[bin_number_xz] += dem_vel.Total_bin_velocity[i]

    dem_vel_av = velocity_list_dem/freq_list_dem

    pept_vel_av = pept_vel_av.to_numpy()
    pept_vel_av = pept_vel_av.flatten()
    dem_vel_av[np.isnan(dem_vel_av)] = -0.05

    compare_grid = np.zeros(len(pept_vel_av))
    cell_counter = 0

    for cell_data in range(len(pept_vel_av)):
        if pept_vel_av[cell_data] > -0.02 and dem_vel_av[cell_data] > -0.02:
            compare_grid[cell_data] = pept_vel_av[cell_data] - dem_vel_av[cell_data]
            cell_counter += 1
    
    error = np.sqrt(np.sum(compare_grid**2)/cell_counter)
    
    return error

error_vsm = simulate_mill(cor,sf,ce_pp,ce_pw,fill)

error = error_vsm



