import numpy as np
from numba import njit
import utils
import calculate as calc

# Generate (x,y,z) coordinates from randomly generated vector in spherical coordinates
@njit
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
    simulation_next = {
        'coordinates': simulation['coordinates'].copy(),
        'patches': [p.copy() for p in simulation['patches']],
        'particles': simulation['particles'], # This usually doesn't change, so reference is okay
        'radii': simulation['radii'],
        'interactions': simulation['interactions']
    }
    simulation_next['patches'] = [p.copy() for p in simulation['patches']]
    simulation_aux = simulation.copy()
    simulation_aux['coordinates'] = simulation['coordinates'].copy()
    simulation_aux['distances'] = simulation['distances'].copy()
    simulation_aux['patches'] = [p.copy() for p in simulation['patches']]
    radii = simulation['radii']
    coords = simulation_next['coordinates']
    n_dims = sim_params['n_dimensions']
    box_limits = sim_params['box_limits']
    n_particles = len(coords)
    cell_size = sim_params['cell_size']
    
    # Building neighbors cell list
    n_cells_xyz = np.array([sim_params['n_cells_1d']]*3, dtype=np.int32)
    head, linked_list = utils.update_cell_list(coords, box_limits, \
                                               cell_size, n_cells_xyz)
    processed = set()
    for i in indexes:
        if i in processed: # If the particle, through clustering, has already been accounted for
            continue
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
                to_process = np.array(to_process, dtype = np.int64)
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
                    rot_matrix = rotation.as_matrix()
                    rotated_coords = rotate_cluster(coords, n_particles, box_limits, n_dims, to_process, rot_matrix)
                    rotated_coords += step
                    clash = False
                    for idx, g_idx in enumerate(to_process):
                        new_pos = rotated_coords[idx]
                        #global_idx = to_process[p_idx_in_cluster]
                        #new_pos = rotated_coords[p_idx_in_cluster]
                        if utils.check_steric_clash_cell(g_idx, new_pos, coords, head, linked_list, radii, box_limits, \
                                                         cell_size, n_cells_xyz, skip_indices = to_process) == True:
                            clash = True
                            break
                    # To define if we accept or reject the step:
                    if clash == False:
                        # Update the current coords array and the linked cell list
                        rotated_patches = rotate_patches(simulation, to_process, rotation)
                        for idx, g_idx in enumerate(to_process):
                            # Update Master local array (for the next cluster in this step to see)
                            coords[g_idx] = rotated_coords[idx]
                            simulation_next['coordinates'][g_idx] = rotated_coords[idx]
                            simulation_next['patches'][g_idx] = rotated_patches[idx]
                            
                        # Rebuild the cell list
                        head, linked_list = utils.update_cell_list(coords, box_limits, cell_size, n_cells_xyz)
                else:
                    sigma = part_params[str(part_type)]['diff_rot']
                    rot_vec = np.random.normal(0, sigma, size=3)
                    rotation = calc.generate_rotation(rot_vec)
                    new_position = coords[i] + step
                    skip_idx = np.array([i], dtype=np.int64)
                    clash = utils.check_steric_clash_cell(i, new_position, coords, head, linked_list, radii, box_limits, \
                                                          cell_size, n_cells_xyz, skip_indices = skip_idx)
                        
                    if clash == True:
                        simulation_next['coordinates'][i] = coords[i]
                    else:
                        coords[i] = new_position
                        simulation_next['coordinates'][i] = new_position
                        rotated_patches = rotate_patches(simulation, to_process, rotation)
                        simulation_next['patches'][i] = rotated_patches[0]
                        head, linked_list = utils.update_cell_list(coords, box_limits, cell_size, n_cells_xyz)
            else:
                new_position = coords[i] + step
                skip_idx = np.array([i], dtype=np.int64)
                clash = utils.check_steric_clash_cell(i, new_position, coords, head, linked_list, radii, box_limits, \
                                                      cell_size, n_cells_xyz, skip_indices = skip_idx)
                if clash == True:
                    simulation_next['coordinates'][i] = coords[i]
                else:
                    coords[i] = new_position
                    simulation_next['coordinates'][i] = new_position
                    head, linked_list = utils.update_cell_list(coords, box_limits, cell_size, n_cells_xyz)
            for item in to_process:
                processed.add(item)
    return simulation_next
            
# =============================================================================
# # Updates distances matrix. Returns updated matrix if no clash exists; returns 0 if there are any clashes
# def update_distances(sim_params, part_params, simulation, to_process = []):
#     new_dists = 1*simulation['distances']
#     check = True
#     for i in to_process:
#         coords_i = simulation['coordinates'][i]
#         type_i = str(simulation['particles'][i])
#         r_hs_i = part_params[type_i]['radius']
#         for j in range(len(simulation['particles'])):
#             if i == j:
#                 pass
#             else:
#                 coords_j = simulation['coordinates'][j]
#                 type_j = str(simulation['particles'][j])
#                 r_hs_j = part_params[type_j]['radius']
#                 r = r_hs_i + r_hs_j
#                 dist = calc.calculate_distance_sq(sim_params['box_limits'], coords_i, coords_j)
#                 new_dists[i,j] = dist
#                 new_dists[j,i] = dist
#                 if new_dists[i,j] <= r**2:
#                     check = False
#                 new_dists[i,j] = np.sqrt(dist)
#                 new_dists[j,i] = new_dists[i,j]
#     return new_dists, check
# =============================================================================

def update_distances(sim_params, part_params, simulation, radii_all, to_process):
    indices = np.array(to_process, dtype = np.int64)
    coords = simulation['coordinates']
    distances = simulation['distances']
    box_limits = sim_params['box_limits']
    n_particles = sim_params['n_particles']
    proceed = update_distances_numba(indices, coords, distances, radii_all, box_limits, n_particles)
    return distances, proceed

# Updates distances matrix. Returns updated matrix if no clash exists; returns 0 if there are any clashes
def update_distances_numba(indices, coords, distances, radii, box_limits, n_particles):
    for i in indices:
        for j in range(n_particles):
            if i == j:
                continue
            dist_sq = calc.calculate_distance_sq(box_limits, coords[i], coords[j])
            j_in_cluster = False
            for p in indices:
                if j == p:
                    j_in_cluster = True
                    break
            if j_in_cluster == False:
                limit = radii[i] + radii[j]
                if dist_sq < limit**2:
                    return False
            dist = np.sqrt(dist_sq)
            distances[i,j] = dist
            distances[j,i] = dist
    return True
    
@njit
def calculate_distance_matrix(coords, box_limits):
    n = len(coords)
    distances = np.zeros((n,n), dtype = np.float64)
    for i in range(n):
        for j in range(i,n):
            if i == j:
                distances[i,j] = 0
            else:
                dist = calc.calculate_distance(box_limits, coords[i], coords[j])
                distances[i,j] = dist
                distances[j,i] = dist
    return distances
    
# Apply periodic boundary conditions where applicable
@njit
def apply_periodic_boundaries(coordinates, box_limits, n_dimensions):
    for i in range(len(coordinates)):
        for d in range(n_dimensions):
            if (np.abs(coordinates[i][d]) > box_limits):
                if coordinates[i][d] > 0:
                    coordinates[i][d] = coordinates[i][d] - 2*box_limits
                else:
                    coordinates[i][d] = coordinates[i][d] + 2*box_limits
            else: 
                pass
    #return simulation['coordinates']
       
# Apply test_interaction for particles within specific distances and with aligned patches
def update_interactions(sim_params, part_params, inter_params, simulation, threshold):
    particle_list = list(range(0,simulation['distances'].shape[0]))
    box_limits = sim_params['box_limits']
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
                            d_vector = calc.calculate_distance_vector_pbc(box_limits, simulation['coordinates'][j], simulation['coordinates'][i])
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
@njit
def get_pbc_neighbor(vec1, vec2, box_limits, n_dimensions):
    new_coords = np.zeros(n_dimensions)
    for i in range(len(vec1)):
        vec1_cands = np.array([vec1[i], vec1[i] + 2*box_limits, vec1[i] - 2*box_limits])
        diff = (vec2[i] - vec1_cands)**2
        min_index = np.argmin(diff)
        new_coords[i] = vec1_cands[min_index]
    return new_coords

# Rotate cluster around center of mass considering periodic boundary conditions
@njit
def rotate_cluster(coordinates, n_particles, box_limits, n_dimensions, processing_list, rotation_matrix):
    ref_index = processing_list[0]
    ref_coords = coordinates[ref_index]
    n_cluster = len(processing_list)
    neighbors = np.zeros((n_cluster, 3))
    neighbors[0] = ref_coords
    for p in range(1, n_cluster):
        neigh_index = processing_list[p]
        point = coordinates[neigh_index]
        neighbors[p] = get_pbc_neighbor(point, ref_coords, box_limits, n_dimensions)
    com = np.zeros(n_dimensions)
    for i in range(n_cluster):
        com += neighbors[i]
    com = com / n_cluster
    shifted_to_origin = neighbors - com
    rotated_coords = np.dot(shifted_to_origin, rotation_matrix.T)
    rotated_coords += com
    return rotated_coords

# Rotate list of patches
def rotate_patches(simulation, processing_list, rotation):
    rotated_patches = []
    for p in range(len(processing_list)):
        part = processing_list[p]
        unrotated_patch = simulation['patches'][part]
        rotated_patch = rotation.apply(np.array(unrotated_patch))
        rotated_patches.append(rotated_patch)
    return rotated_patches