import math
import numpy as np
import jax.numpy as jnp
from plotfn import PlotEnv
from time import time 
from mppi_fn import MPPIControllerForPathTracking
from jax import lax, random
from npy_fn import Mppi_npy
from matplotlib import pyplot as plt
import jax



def run_simulation_mppi_pathtracking() -> None:
    """run simulation of pathtracking with MPPI controller"""
    print("[INFO] Start simulation of pathtracking with MPPI controller")

    # simulation settings
    delta_t = 0.05 # [sec]
    sim_steps = 10000 # [steps]
    print(f"[INFO] delta_t : {delta_t:.2f}[s] , sim_steps : {sim_steps}[steps], total_sim_time : {delta_t*sim_steps:.2f}[s]")

    # load the reference path
    ref_path = np.genfromtxt('ovalpath.csv', delimiter=',', skip_header=1)
    
    plotter = PlotEnv(ref_path,'r')

    max_omega = 0.785
    max_vel = 2.0
    T = 50  #even number
    # initialize a vehicle as a control target
    init_state = jnp.array([0.0, 0.0, 0.0]) # [x[m], y[m], yaw[rad]]
    current_state = jnp.copy(init_state)
    # initialize a mppi controller for the vehicle
    mppi = MPPIControllerForPathTracking(
        delta_t = delta_t*2.0, # [s]
        max_omega = max_omega, # [rad]      
        max_vel = max_vel, # [m/s^2]
        horizon_step_T = T, # [steps]
        number_of_samples_K = 100, #500 [samples]
        param_exploration = 0.0,
        param_lambda = 100.0,
        param_alpha = 0.98,
        sigma = jnp.diag(jnp.array([0.0075,0.5])),
        stage_cost_weight = jnp.array([50.0, 50.0, 50.0, 1.0]), # weight for [x, y, yaw, v]
        terminal_cost_weight = jnp.array([50.0, 50.0, 50.0, 1.0]), # weight for [x, y, yaw, v]
        window_size= T
    )

    npy_ob = Mppi_npy(np.array(ref_path),T)

    u_prev = jnp.zeros((T, 2))

    key = random.PRNGKey(0)
    # simulation loop
    for i in range(sim_steps):
        key, subkey = random.split(key)
        start_time = time()
        ref_x, ref_y, ref_yaw, ref_v = npy_ob.get_nearest_waypoint(np.asarray(current_state[0]),np.asarray(current_state[1]))
        # print(ref_y)
        # print(ref_yaw)
        # exit(0)
        ref_x = jnp.asarray(ref_x)
        ref_y = jnp.asarray(ref_y)
        ref_yaw = jnp.asarray(ref_yaw)
        ref_v = jnp.asarray(ref_v)
        # calculate input force with MPPI
        optimal_input, optimal_input_sequence, optimal_traj, sampled_traj_list, u_prev = mppi.calc_control_input(current_state, ref_x, ref_y,\
                                                                                                                    ref_yaw, ref_v, key, u_prev)
        print("elapsed=",time()-start_time)
        # print(optimal_input)
        # print(mppi.u_prev)
        # exit(0)
        # print current state and input force
        print(f"Time: {i*delta_t:>2.2f}[s], x={current_state[0]:>+3.3f}[m], y={current_state[1]:>+3.3f}[m], yaw={current_state[2]:>+3.3f}[rad], omega={optimal_input[0]:>+6.2f}[rad/s], vel={optimal_input[1]:>+6.2f}[m/s]")

        optimal_traj_new = optimal_traj[:, 0:2]
        sampled_traj_list_new = sampled_traj_list[:, :, 0:2]
    
        # limit control inputs
        omega = jnp.clip(optimal_input[0],-max_omega,max_omega)
        v = jnp.clip(optimal_input[1],-max_vel,max_vel)

        current_state = mppi.rk4(current_state,jnp.array([omega,v]))

        plotter.step(np.asarray(current_state),np.asarray(sampled_traj_list_new),np.asarray(optimal_traj_new))

if __name__ == "__main__":
    run_simulation_mppi_pathtracking()
    plt.ioff()
    plt.show()
