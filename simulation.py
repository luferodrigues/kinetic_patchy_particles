import os
import numpy as np
import steps as step
import utils
import initialize as init
import calculate as calc

# Run one simulation step
def simulation_step(sim_params, part_params, inter_params, simulation, threshold):
    simulation_next = step.generate_next_step(sim_params, part_params, simulation)
    simulation_next['distances'] = step.calculate_distance_matrix(simulation_next['coordinates'], sim_params['box_limits'])
    step.apply_periodic_boundaries(simulation_next['coordinates'], sim_params['box_limits'], sim_params['n_dimensions'])
    simulation_next['particles'] = simulation['particles']
    # Copying n_bonds from the previous step to run update_interactions, and the I'll update the number of bonds!
    simulation_next['n_bonds'] = simulation['n_bonds']
    # We need to update the interactions, clusters and n_bonds now
    simulation_next['interactions'] = step.update_interactions(sim_params, part_params, inter_params, simulation_next, threshold)
    simulation_next['clusters'] = step.update_clusters(simulation_next['interactions'])
    simulation_next['n_bonds'] = utils.find_bonds_all(part_params, simulation)
    return simulation_next

# Initialize and run N simulation steps
def run_simulation(sim_params, part_params, inter_params, prefix = '', folder = '', transitions = True, clusters_steps = True, manual_dist = 0):
    labels_com = ['C', 'N', 'O', 'F', 'B', 'Ne', 'Al', 'Si', 'P', 'S', 'Cl',
                 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
                 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'UUt', 'Fl', 'Uup', 'Lv', 'Uus', 'Uuo']
    labels_patch = ['H', 'He', 'Li', 'Be', 'Na', 'Mg', 
                     'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
                     'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 
                     'Cs', 'Ba', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 
                     'Fr', 'Ra', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn']
    simulation = {}
    path_coords = os.path.join(folder, prefix, 'trajectory.csv')
    path_xyz = os.path.join(folder, prefix, 'visualization.xyz')
    path_vmd = os.path.join(folder, prefix, 'vmd_init.tcl')
    #path_dists = os.path.join(folder, prefix, 'distances.csv')
    path_clusters = os.path.join(folder, prefix, 'clusters.csv')
    path_cluster_max = os.path.join(folder, prefix, 'cluster_max.csv')
    path_cluster_n = os.path.join(folder, prefix, 'cluster_n.csv')
    path_cluster_agg = os.path.join(folder, prefix, 'cluster_agg.csv')
    path_particles = os.path.join(folder, prefix, 'particles.csv')
    path_n_bonds = os.path.join(folder, prefix, 'n_bonds.csv')
    path_transitions = os.path.join(folder, prefix, 'transitions.csv')
    # Initializing elements
    coordinates = init.initialize_coordinates(sim_params, part_params)
    distances = init.initialize_distances(sim_params)
    interactions = init.initialize_interactions(sim_params, part_params)
    clusters = init.initialize_clusters()
    particles = init.initialize_particles(part_params)
    particle_types = particles.astype(np.int64)
    # Group everything in a single dictionary
    simulation = {
        'coordinates': coordinates,
        'distances': distances,
        'interactions': interactions,
        'clusters': clusters,
        'particles': particles
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
    n_bonds = init.initialize_n_bonds(patches)
    simulation['n_bonds'] = n_bonds
    radii_types = np.array([part_params[str(t)]['radius'] for t in range(len(part_params))])
    radii_all = radii_types[particle_types]
    simulation['radii'] = radii_all
    
    cell_size = threshold # Setting length for neighbor cells (could be a multiplication of threshold, for example)
    n_cells_1d = int(np.floor((2 * sim_params['box_limits']) / cell_size)) # Finding the number of cells
    sim_params['n_cells_1d'] = n_cells_1d
    sim_params['cell_size'] = (2 * sim_params['box_limits']) / n_cells_1d
    
    for i in range(sim_params['n_particles']):
        process = [i]
        distances, proceed = step.update_distances(sim_params, part_params, simulation, radii_all, process)
        # Check to avoid steric clashes when initializing system
        try:
            if proceed == False:
                while proceed == False:
                    simulation['coordinates'][i] = np.random.uniform(low = -sim_params['box_limits'], high = sim_params['box_limits'], \
                                                          size = sim_params['n_dimensions'])
                    distances, proceed = step.update_distances(sim_params, part_params, simulation, radii_all, process)
                simulation['distances'] = distances
        except:        
            simulation['distances'] = distances
    simulation['interactions'] = step.update_interactions(sim_params, part_params, inter_params, simulation, threshold)
    simulation['clusters'] = step.update_clusters(simulation['interactions'])
    simulation['n_bonds'] = utils.find_bonds_all(part_params, simulation)
    
    # Setting up dictionaries for .xyz files
    labels_part_atom, labels_part_patch, labels_patch_atom = \
        utils.build_label_dictionaries(part_params, labels_com, labels_patch)
    # Write tcl file for visualization
    utils.write_tcl(path_vmd, sim_params, part_params, labels_part_atom, labels_part_patch, labels_patch_atom)
    
    # -------------------------------- #
   
    print('----------------')
    
    simulation_handle = [simulation, simulation.copy()]
    new_file_toggle = True
    new_file_transitions_toggle = True
    new_file_bonds_toggle = True
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
        clusters_previous = simulation_handle[previous]['clusters']
        simulation_this_step = simulation_step(sim_params, part_params, inter_params, simulation_handle[previous], threshold)
        simulation_this_step['current_step'] = i
        simulation_handle[current] = simulation_this_step
        n_bonds = simulation_handle[current]['n_bonds']
        clusters = simulation_handle[current]['clusters']
        cluster_number, cluster_agg, n_max = utils.obtain_cluster_features(sim_params, clusters)
        # After the first step we can look for transitions, which are recorded at every step
        if (i > 1) and (len(clusters) > 0) and (transitions == True):
            clusters_tracking_previous = utils.clusters_origins_reference(clusters_previous)
            clusters_tracking_current = utils.find_cluster_origins(simulation_this_step, clusters_previous)
            transitions_matrix = utils.generate_transitions_matrix(clusters_tracking_previous, clusters_tracking_current)
            cluster_transitions = utils.assign_transitions(transitions_matrix)
            utils.write_transitions(path_transitions, cluster_transitions, frame_number = i, new_file = new_file_transitions_toggle)
            new_file_transitions_toggle = False
            if clusters_steps == True:
                utils.write_clusters(path_clusters, clusters, frame_number = i, new_file = new_file_toggle)
        if (i == 1) or (i%sim_params['n_interval'] == 0):
            utils.write_coords_csv(path_coords, sim_params, part_params, simulation_handle[current], frame_number=i, new_file = new_file_toggle)
            utils.write_coords_xyz(path_xyz, sim_params, part_params, simulation_handle[current], 
                                   labels_part_atom, labels_part_patch, labels_patch_atom, frame_number=i, 
                                   new_file = new_file_toggle)
            #utils.write_matrix(path_dists, sim_params, simulation_this_step['distances'], frame_number=i, new_file = new_file_toggle)
            if clusters_steps == False:
                utils.write_clusters(path_clusters, clusters, frame_number = i, new_file = new_file_toggle)
            utils.write_clusters_feats(path_cluster_agg, cluster_agg, new_file = new_file_toggle)
            utils.write_clusters_int(path_cluster_n, cluster_number, new_file = new_file_toggle)
            utils.write_clusters_int(path_cluster_max, n_max, new_file = new_file_toggle)
            for cluster in clusters:
                for particle_index in cluster:
                    utils.write_n_bonds(path_n_bonds, particle_index, n_bonds[particle_index], frame_number=i, new_file = new_file_bonds_toggle)
                    new_file_bonds_toggle = False
            if new_file_toggle == True:
                new_file_toggle = False
        else:
            pass
    print('\n----------------')
    print('Done!')
    return simulation_this_step
