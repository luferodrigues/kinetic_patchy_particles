import numpy as np
import utils
import calculate as calc

# Generate (x,y,z) coordinates from randomly generated vector in spherical coordinates
def generate_step_coords(diff, dt = 1):
    sigma = np.sqrt(2*diff*dt)
    r = np.abs(np.random.normal(loc = 0, scale = sigma))
    theta = np.random.uniform(low = 0, high = np.pi)
    phi = np.random.uniform(low = 0, high = 2*np.pi)
    step_coords = calc.spherical_to_cartesian(r, theta, phi) # List with coordinates [x, y, z]
    return step_coords

# Generate next step from gaussian distributions. To account for hard spheres and different diffusion coefficients, we do particle by particle
def generate_next_step(sim_params, part_params, simulation, verbose = False):
    indexes = np.arange(sim_params['n_particles'])
    # Randomizing list of indices for every step so no particle has "priority" throughout the simulation
    np.random.shuffle(indexes)
    simulation_next = simulation.copy()
    simulation_next['coordinates'] = simulation['coordinates'].copy()
    simulation_next['distances'] = simulation['distances'].copy()
    simulation_next['patches'] = [p.copy() for p in simulation['patches']]
    simulation_aux = simulation.copy()
    simulation_aux['coordinates'] = simulation['coordinates'].copy()
    simulation_aux['distances'] = simulation['distances'].copy()
    simulation_aux['patches'] = [p.copy() for p in simulation['patches']]
    processed = []
    for i in indexes:
        if i in processed: # If the particle, through clustering, has already been accounted for
            pass
        else:
            to_process = [i]
            part_type = int(simulation['particles'][i])
            diff = part_params[str(part_type)]['diff_trans']
            interacting = part_params[str(part_type)]['interacting']
            if interacting == False:
                pass
            else:
                found, cluster_index, particle_index = utils.search_sublists(simulation['clusters'], i)
                if found == True: # If it is an interacting particle in a cluster
                    cluster_i = simulation['clusters'][cluster_index]
                    for j in range(len(cluster_i)):
                        to_process.append(cluster_i[j])
                    del to_process[0]
                else: 
                    pass
            # If cluster, calculate diffusion coefficient
            if ( len(to_process) > 1 ):
                diff = calc.diffusion_cluster(part_params, simulation, to_process, which = 'trans')
                diff_rot = calc.diffusion_cluster(part_params, simulation, to_process, which = 'rot')
            else:
                pass
            step = generate_step_coords(diff, sim_params['time_step'])
            if ( interacting == True ):
                if ( len(to_process) > 1 ):
                    # Randomize rotation axis, calculate CoM, rotate all to_process particles
                    sigma_rot = calc.diff_to_sigma(diff_rot, sim_params['time_step'])
                    rot_vec = np.random.normal(0, sigma_rot, size=3)
                    rotation = calc.generate_rotation(rot_vec)
                    rotated_coords, rot_order = rotate_cluster(sim_params, simulation, to_process, rotation)
                    rotated_coords += step
                    # JUNTAR ROTATED COORDS A UM COORDS EXTERNO PRA ATUALIZAR DISTÂNCIAS
                    counter = 0
                    for ro in rot_order:
                        simulation_aux['coordinates'][ro] = rotated_coords[counter]
                        counter += 1
                    new_distances, proceed = update_distances(sim_params, part_params, simulation_aux, to_process = to_process)
                    if proceed == False:
                        pass
                    else:
                        rotated_patches = rotate_patches(simulation, to_process, rotation)
                        counter = 0
                        for ro in rot_order:
                            simulation_next['coordinates'][ro] = rotated_coords[counter]
                            counter += 1
                        simulation_next['distances'] = new_distances
                        counter = 0
                        for tp in to_process:
                            simulation_next['patches'][tp] = rotated_patches[counter]
                            counter += 1
                else:
                    sigma = part_params[str(part_type)]['diff_rot']
                    rot_vec = np.random.normal(0, sigma, size=3)
                    rotation = calc.generate_rotation(rot_vec)
                    simulation_aux['coordinates'][i] = simulation['coordinates'][i] + step
                    new_distances, proceed = update_distances(sim_params, part_params, simulation_aux, to_process = to_process)
                    if proceed == False:
                        simulation_next['coordinates'][i] = simulation['coordinates'][i]
                    else:
                        simulation_next['coordinates'][i] = simulation_aux['coordinates'][i]
                        simulation_next['distances'] = new_distances
                        rotated_patches = rotate_patches(simulation, to_process, rotation)
                        simulation_next['patches'][i] = rotated_patches[0]
            else:
                simulation_aux['coordinates'][i] = simulation['coordinates'][i] + step
                new_distances, proceed = update_distances(sim_params, part_params, simulation_aux, to_process = to_process)
                if proceed == False:
                    simulation_next['coordinates'][i] = simulation['coordinates'][i]
                else:
                    simulation_next['coordinates'][i] = simulation_aux['coordinates'][i]
                    simulation_next['distances'] = new_distances
            for item in to_process:
                processed.append(item)
    return simulation_next
            
# Updates distances matrix. Returns updated matrix if no clash exists; returns 0 if there are any clashes
def update_distances(sim_params, part_params, simulation, to_process = []):
    new_dists = 1*simulation['distances']
    check = True
    for i in to_process:
        for j in range(len(simulation['particles'])):
            if i == j:
                pass
            else:
                coords_i = simulation['coordinates'][i]
                coords_j = simulation['coordinates'][j]
                type_i = str(simulation['particles'][i])
                type_j = str(simulation['particles'][j])
                r_hs_i = part_params[type_i]['radius']
                r_hs_j = part_params[type_j]['radius']
                r = r_hs_i + r_hs_j
                dist = calc.calculate_distance(sim_params, coords_i, coords_j)
                new_dists[i,j] = dist
                new_dists[j,i] = dist
                if new_dists[i,j] <= r:
                    check = False
    return new_dists, check

# Apply periodic boundary conditions where applicable
def apply_periodic_boundaries(sim_params, simulation):
    for i in range(len(simulation['coordinates'])):
        for d in range(sim_params['n_dimensions']):
            if (np.abs(simulation['coordinates'][i][d]) > sim_params['box_limits']):
                if simulation['coordinates'][i][d] > 0:
                    simulation['coordinates'][i][d] = simulation['coordinates'][i][d] - 2*sim_params['box_limits']
                else:
                    simulation['coordinates'][i][d] = simulation['coordinates'][i][d] + 2*sim_params['box_limits']
            else: 
                pass
    #return simulation['coordinates']
       
# Apply test_interaction for particles within specific distances and with aligned patches
def update_interactions(part_params, inter_params, simulation, threshold):
    particle_list = list(range(0,simulation['distances'].shape[0]))
    interactions_new = 1 * simulation['interactions']
    for i in range(0, len(particle_list)):
        if len(particle_list) == 1:
            pass
        else:
            for j in range(particle_list[0], particle_list[-1]+1):
                distance12 = simulation['distances'][i][j]
                # Checks if it's close enough to start verifying the distance
                if distance12 <= threshold:
                    particle1 = simulation['particles'][i]
                    particle2 = simulation['particles'][j]
                    params1 = utils.get_dict_params(part_params, str(particle1))
                    params2 = utils.get_dict_params(part_params, str(particle2))
                    # Checks if they are interacting particles by themselves
                    if (params1['interacting'] == True) and (params2['interacting'] == True):
                        patches1 = simulation['patches'][i]
                        patches2 = simulation['patches'][j]
                        #inter = utils.get_dict_params(inter_params, (particle1, particle2))
                        # Sorts particle types in ascending order
                        if particle1 < particle2:
                            pair_particles = (particle1, particle2)
                        else:
                            pair_particles = (particle2, particle1)
                        compatible = utils.check_particle_interaction(inter_params, particle1, particle2)
                        # Checks if both particles can interact
                        if compatible == True:
                            # Compatible if some pair of patches are pointing at each other
                            alphas1 = params1['patches']['alphas']
                            alphas2 = params2['patches']['alphas']
                            d_vector = simulation['coordinates'][i] - simulation['coordinates'][j]
                            orient_check = utils.check_alignment_all(patches1, patches2, alphas1, alphas2, d_vector)
                            aligned = orient_check[0]
                            pair_patches = orient_check[1]
                            # Checks if there are aligned patches
                            if aligned == True:
                                a = pair_patches[0]
                                b = pair_patches[1]
                                r_hs1 = params1['radius']
                                r_hs2 = params2['radius']
                                r_p1 = params1['patches']['radius'][a]
                                r_p2 = params2['patches']['radius'][b]
                                r_limit = r_hs1 + r_p1 + r_hs2 + r_p2
                                dist_check = utils.check_distance(r_limit, distance12)
                                # Checks if oriented patches are within intraction distance
                                if dist_check == True:
                                    inter_ij = simulation['interactions'][i][j]
                                    patch_type1 = pair_patches[0]
                                    patch_type2 = pair_patches[1]
                                    has_negative = np.any(inter_ij < 0)
                                    # Checks if they are interacting
                                    if has_negative == True:
                                        prob = inter_params[pair_particles]['p_ass'][patch_type1][patch_type2]
                                        interactions_new[i][j] = utils.test_association(patch_type1, patch_type2, prob)
                                    else:
                                        prob = inter_params[pair_particles]['p_diss'][patch_type1][patch_type2]
                                        interactions_new[i][j] = utils.test_dissociation(patch_type1, patch_type2, prob)
                        interactions_new[j][i] = interactions_new[i][j]
            particle_list.remove(particle_list[0])
        # Force diagonal terms to be non-interacting (sometimes they were being set to interacting, not sure why)
        for k in range(0,2):
            interactions_new[i][i][k] = -1
    return interactions_new
        
# Converts from patch interactions matrix to boolean particle-particle interaction matrix
def interactions_patches_to_boolean(interactions):
    boolean = np.all(interactions >= 0, axis=2)
    return boolean

def update_clusters(interactions):
    n = interactions.shape[0]
    visited = set()
    clusters = []
    bonded = interactions_patches_to_boolean(interactions)
    for i in range(n):
        if i in visited:
            continue
        stack = [i]
        cluster = []
        while len(stack) > 0:
            p = stack.pop()
            if p in visited:
                continue
            visited.add(p)
            cluster.append(p)
            neighbors = np.where(bonded[p])[0]
            for n in neighbors:
                if n not in visited:
                    stack.append(n)
        if len(cluster) > 1:
            clusters.append(cluster)
    return clusters

# =============================================================================
# def eject_from_cluster(index_separation, interactions_bool, coordinates, diff):
#     clusters = update_clusters(interactions_bool)
#     target_cluster = # Search
#     accept = False
#     while accept == False:
#         coord_move = generate_step_coords(diff)
#         for index in target_cluster:
#             candidate_coordinate = coordinates[index]
#         
#         # MOVE THE WHOLE CLUSTER TOGETHER
#     # TRY TO MOVE A "SAFE DISTANCE AWAY" (DON'Y TRY JUST A SINGLE MOVE)
#     return 0
# =============================================================================

# Find coordinate from potentially neighboring (periodic) boxes with minimum distance to reference vector (vec2)
def get_pbc_neighbor(sim_params, vec1, vec2):
    new_coords = np.zeros(sim_params['n_dimensions'])
    for i in range(len(vec1)):
        vec1_cands = np.array([vec1[i], vec1[i] + 2*sim_params['box_limits'], vec1[i] - 2*sim_params['box_limits']])
        diff = (vec2[i] - vec1_cands)**2
        min_index = np.argmin(diff)
        new_coords[i] = vec1_cands[min_index]
    return new_coords

# Rotate cluster around center of mass considering periodic boundary conditions
def rotate_cluster(sim_params, simulation, processing_list, rotation):
    ref_index = processing_list[0]
    ref_coords = simulation['coordinates'][ref_index]
    neighbors = [ref_coords]
    for p in range(1, len(processing_list)):
        neigh_index = processing_list[p]
        point = simulation['coordinates'][neigh_index]
        neigh = get_pbc_neighbor(sim_params, point, ref_coords)
        neighbors.append(neigh)
    neighbors = np.array(neighbors)
    com = calc.center_of_mass(neighbors)
    rotated_coords = rotation.apply(neighbors - com)
    rotated_coords += com
    return rotated_coords, processing_list

# Rotate list of patches
def rotate_patches(simulation, processing_list, rotation):
    rotated_patches = []
    for p in range(len(processing_list)):
        part = processing_list[p]
        unrotated_patch = simulation['patches'][part]
        rotated_patch = rotation.apply(np.array(unrotated_patch))
        rotated_patches.append(rotated_patch)
    return rotated_patches