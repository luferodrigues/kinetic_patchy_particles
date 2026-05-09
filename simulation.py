import os
import numpy as np
import steps as step
import utils
import initialize as init
import calculate as calc

# Run one simulation step
def simulation_step(sim_params, part_params, inter_params, simulation, threshold):
    simulation_next = {}
    simulation_next = step.generate_next_step(sim_params, part_params, simulation)
    step.apply_periodic_boundaries(sim_params, simulation_next)
    simulation_next['particles'] = simulation['particles']
    # Until here we have updated only the coordinates of CoM coords, patch orientations and distance. 
    # We need to update the interactions and clusters now
    for i in range(sim_params['n_particles']):
        process = [i]
        distances, proceed = step.update_distances(sim_params, part_params, simulation_next, process)
        simulation_next['distances'] = distances
    simulation_next['interactions'] = step.update_interactions(part_params, inter_params, simulation_next, threshold)
    simulation_next['clusters'] = step.update_clusters(simulation_next['interactions'])
    return simulation_next

# Initialize and run N simulation steps
def run_simulation(sim_params, part_params, inter_params, prefix = '', folder = '', manual_dist = 0):
    simulation = {}
    path_coords = os.path.join(folder, prefix, 'trajectory.csv')
    path_xyz = os.path.join(folder, prefix, 'visualization.xyz')
    path_dists = os.path.join(folder, prefix, 'distances.csv')
    path_cluster_max = os.path.join(folder, prefix, 'cluster_max.csv')
    path_cluster_n = os.path.join(folder, prefix, 'cluster_n.csv')
    path_cluster_agg = os.path.join(folder, prefix, 'cluster_agg.csv')
    path_particles = os.path.join(folder, prefix, 'particles.csv')
    # Initializing elements
    coordinates = init.initialize_coordinates(sim_params, part_params)
    distances = init.initialize_distances(sim_params)
    interactions = init.initialize_interactions(sim_params, part_params)
    clusters = init.initialize_clusters()
    particles = init.initialize_particles(part_params)
    # Group everything in a single dictionary
    simulation = {
        'coordinates': coordinates,
        'distances': distances,
        'interactions': interactions,
        'clusters': clusters,
        'particles': particles,
        }
    utils.write_particle_types(path_particles, simulation)
    for key in part_params.keys():
        dt = sim_params['time_step']
        try:
            diff_trans = part_params[key]['diff_trans']
        except:
            visc = sim_params['viscosity']
            temp = sim_params['temperature']
            r_hs = part_params[key]['radius']
            diff_trans = calc.calculate_diff_trans(r_hs, viscosity = visc, temperature = temp)
            part_params[key]['diff_trans'] = diff_trans
        part_params[key]['sigma_trans'] = calc.diff_to_sigma(diff_trans, dt)
        if part_params[key]['interacting'] == True:
            try:
                diff_rot = part_params[key]['diff_rot']
            except:
                visc = sim_params['viscosity']
                temp = sim_params['temperature']
                r_hs = part_params[key]['radius']
                diff_rot = calc.calculate_diff_rotation(r_hs, viscosity = visc, temperature = temp)
                part_params[key]['diff_rot'] = diff_rot
            part_params[key]['sigma_rot'] = calc.diff_to_sigma(diff_rot, dt)
    # Find/set minimum distance to verify distance and patch alignment
    threshold = utils.max_distance(part_params, manual_dist)
    patches = init.initialize_patches(sim_params, part_params, simulation)
    simulation['patches'] = patches
    for i in range(sim_params['n_particles']):
        process = [i]
        distances, proceed = step.update_distances(sim_params, part_params, simulation, process)
        # Check to avoid steric clashes when initializing system
        try:
            if proceed == False:
                while proceed == False:
                    simulation['coordinates'][i] = np.random.uniform(low = -sim_params['box_limits'], high = sim_params['box_limits'], \
                                                          size = sim_params['n_dimensions'])
                    distances, proceed = step.update_distances(sim_params, part_params, simulation, process)
                simulation['distances'] = distances
        except:        
            simulation['distances'] = distances
    simulation['interactions'] = step.update_interactions(part_params, inter_params, simulation, threshold)
    simulation['clusters'] = step.update_clusters(simulation['interactions'])
    
    # -------------------------------- #
   
    print('----------------')
    
    simulation_handle = [simulation, simulation.copy()]
    new_file_toggle = True
    for i in range(1, sim_params['n_steps']+1):
        if i%(sim_params['n_steps'] / 100) == 0:
            print(f"Progress: {int(i/sim_params['n_steps']*100)}%", end = '\r')
        current = i % 2
        if current == 0:
            previous = 1
        elif current == 1:
            previous = 0
        else:
            pass
        simulation_this_step = simulation_step(sim_params, part_params, inter_params, simulation_handle[previous], threshold)
        simulation_this_step['current_step'] = i
        simulation_handle[current] = simulation_this_step
        clusters = simulation_handle[current]['clusters']
        cluster_number, cluster_agg, n_max = utils.obtain_cluster_features(sim_params, clusters)
        if (i == 1) or (i%sim_params['n_interval'] == 0):
            utils.write_coords_csv(path_coords, sim_params, part_params, simulation_handle[current], frame_number=i, new_file = new_file_toggle)
            utils.write_coords_xyz(path_xyz, sim_params, part_params, simulation_handle[current], frame_number=i, new_file = new_file_toggle)
            utils.write_matrix(path_dists, sim_params, simulation_this_step['distances'], frame_number=i, new_file = new_file_toggle)
            utils.write_clusters_feats(path_cluster_agg, cluster_agg, new_file = new_file_toggle)
            utils.write_clusters_int(path_cluster_n, cluster_number, new_file = new_file_toggle)
            utils.write_clusters_int(path_cluster_max, n_max, new_file = new_file_toggle)
            if (i == 1):
                new_file_toggle = False
        else:
            pass
    print('\n----------------')
    print('Done!')
    return simulation_this_step
