import os
import numpy as np
from numba import njit
import calculate as calc

# Find target in sublists within a list
def search_sublists(list_of_lists, target):
    found = False
    found_target = -1
    found_list = -1
    for sublist in list_of_lists:
        if target in sublist:
            found_list = list_of_lists.index(sublist)
            found_target = sublist.index(target)
            found = True
            break  # Stop searching after finding the first occurrence
    return found, found_list, found_target

# Merge list of lists which has common elements into a smaller list of lists
def merge_lists(list_of_lists):
    l = list_of_lists
    out = []
    while len(l)>0:
        first, rest = l[0], l[1:]
        first = set(first)
        lf = -1 # Valor meio dummy
        while len(first) > lf:
            lf = len(first)
            rest2 = []
            for r in rest:
                if len(first.intersection(set(r)))>0:
                    first |= set(r)
                else:
                    rest2.append(r)     
            rest = rest2
        out.append(first)
        l = rest
    out_list = [list(s) for s in out] # Converte a lista de sets para lista de listas
    return out_list

# "Correct" numpy histogram: placing distribution between edges
def correct_histogram(histogram):
    bins_centers = histogram[1][:-1] + (histogram[1][1] - histogram[1][0]) / 2 # Exclude last point and add half REGULAR bin width
    return [bins_centers, histogram[0]]

# Generate parameters file with inputs
def generate_parameters_file(parameters_dictionary, folder = '', prefix = ''):
    out = os.path.join(folder, prefix + '_' + 'parameters.par')
    with open(out, 'w') as f:
        for key, value in parameters_dictionary.items():
            f.write(str(key) + ',' + str(value))
            f.write('\n')
    
# Import file
def import_file(file, which = 'data', path = ''):
    data = []
    handle = []
    which_types = ['trajectory', 'clusters', 'interactions', 'distances', 'particles', 'cluster_number', 'cluster_agg']
    if which in which_types:
        with open(file, 'r') as input_file:
            if which in ['trajectory', 'clusters', 'interactions', 'distances']:
                for line in input_file:
                    entry = line.split(',')
                    if '\n' not in entry[0]:
                        aux = []
                        for i in range(len(entry) - 1):
                            if which in ['clusters', 'interactions']:
                                floating = float(entry[i])
                                aux.append(int(floating))
                            else:
                                aux.append(float(entry[i]))
                        handle.append(aux)
                    else:
                        if entry[0] == '\n':
                            data.append(handle)
                            handle = []
                        else:
                            pass                    
            elif which in ['particles', 'cluster_number']:
                for line in input_file:
                    remove_n = '\n'
                    line_new = line.replace(remove_n, '')
                    data.append(int(line_new))
            elif which == 'cluster_agg':
                for line in input_file:
                    entry = line.split(',')
                    for i in range(len(entry)):
                        if entry[i] == '\n':
                            pass
                        else:
                            handle.append(int(entry[i]))
                    data.append(handle)
                    handle = []
            else:
                    pass
    else:
        print("Please select compatible 'which' parameter")
    return data

# Count total number of particles in simulation
def count_particles(particles_params):
    n = 0
    for key, val in particles_params.items():
        for k, v in particles_params[key].items():
            if k == 'number':
                n += particles_params[key][k]
    return n

# -------------------------------
# ----IMPORT PARAMETERS BLOCK----
# -------------------------------

# =============================================================================
# def import_particle_params(filename):
#     particle_parameters = {}
#     new_parameter = 'dummy'
#     multi_line = False
#     start_new = False
#     particle_number = 0
#     particle_parameters[str(particle_number)] = {}
#     new_list = []
#     with open(filename, 'r') as file:
#         for line in file:
#             l = line.strip()
#             l_split = l.split(',')
# 
#             if len(l_split) <= 2:
#                 if len(new_list) == 0:
#                     pass
#                 elif len(new_list) == 1:
#                     particle_parameters[str(particle_number)][new_parameter] = new_list[0]
#                 else:
#                     particle_parameters[str(particle_number)][new_parameter] = new_list
#                 multi_line = False
#                 new_list = []
#                 if l_split[0] == '':
#                     start_new = True
#                     particle_number += 1
#                 # Prepare next parameter entry
#                 else:
#                     if start_new == True:
#                         particle_parameters[str(particle_number)] = {}
#                         start_new = False
#                     else:
#                         pass
#                     if l_split[0].lower() == 'yes' or l_split[0].lower() == 'true':
#                         particle_parameters[str(particle_number)]['interacting'] = True
#                     elif l_split[0].lower() == 'no' or l_split[0].lower() == 'false':
#                         particle_parameters[str(particle_number)]['interacting'] = False
#                     try:
#                         particle_parameters[str(particle_number)][new_parameter] = float(l_split[0])
#                     except:
#                         new_parameter = l_split[0]
#             else:
#                 multi_line = True
#                 mini_list = []
#                 for i in range(len(l_split)-1):
#                     mini_list.append(float(l_split[i]))
#                 new_list.append(mini_list)
# 
#     for key, val in particle_parameters.items():
#         for k, v in particle_parameters[key].items():
#             if k == 'type' or k == 'number':
#                 particle_parameters[key][k] = int(v)
#             else:
#                 pass
#     
#     grouped_particle_parameters = group_patch_params(particle_parameters)
#     return grouped_particle_parameters
# =============================================================================


def import_particle_params(filename):
    list_params = {"patches_radius", "patches_alphas"}
    multiline_params = {"patches_positions"}
    particle_parameters = {}
    particle_number = 0
    particle_parameters[str(particle_number)] = {}
    current_param = None
    with open(filename, "r") as f:
        lines = [line.strip() for line in f]
        
    i = 0
    while i < len(lines):
        line = lines[i]
        # Blank line indicates a new particle
        if line == "":
            particle_number += 1
            if i < len(lines) - 1:
                particle_parameters[str(particle_number)] = {}
            i += 1
            continue
        current_param = line
        i += 1
        if i >= len(lines):
            break
        if current_param in multiline_params:
            values = []
            while i < len(lines):
                l = lines[i]
                # End inner loop when ending particle
                if l == "":
                    break
                # End inner loop when next parameter name is reached
                if "," not in l:
                    break
                row = [float(x) for x in l.split(",")[:-1]]
                values.append(row)
                i += 1
            particle_parameters[str(particle_number)][current_param] = values
            continue
        # Single line value
        l = lines[i]
        tokens = l.split(",")[:-1]
        if current_param == "interacting":
            # Already converts to boolean!
            particle_parameters[str(particle_number)][current_param] = (tokens[0].lower() in ("yes", "true"))
        elif current_param in list_params:
            particle_parameters[str(particle_number)][current_param] = [float(x) for x in tokens]
        elif current_param in {"type", "number"}:
            particle_parameters[str(particle_number)][current_param] = int(tokens[0])
        elif current_param == "radius":
            particle_parameters[str(particle_number)][current_param] = float(tokens[0])
        else:
            # Name or any future string parameter
            try:
                particle_parameters[str(particle_number)][current_param] = float(tokens[0])
            except ValueError:
                particle_parameters[str(particle_number)][current_param] = tokens[0]
        i += 1
    return group_patch_params(particle_parameters)


# =============================================================================
# def import_interaction_params(filename):
#     interaction_parameters = {}
#     new_parameter = 'dummy'
#     multi_line = False
#     start_new = False
#     new_interaction = False
#     new_list = []
#     type0 = 0
#     type1 = 0
#     with open(filename, 'r') as file:
#         for line in file:
#             l = line.strip()
#             l_split = l.split(',')
#             
#             if l_split[0] == 'type':
#                 new_interaction = True
#             else:
#                 if new_interaction == True:
#                     type0 = int(float(l_split[0]))
#                     type1 = int(float(l_split[1]))
#                     interaction_parameters[(type0, type1)] = {}
#                     new_interaction = False
#                 else:
#                     if len(l_split) <= 2:
#                         if len(new_list) == 0:
#                             pass
#                         elif len(new_list) == 1:
#                             interaction_parameters[(type0, type1)][new_parameter] = new_list[0]
#                         else:
#                             interaction_parameters[(type0, type1)][new_parameter] = new_list
#                         if len(l_split) == 1:
#                             new_parameter = l_split[0]
#                         else:
#                             pass
#                         multi_line = False
#                         new_list = []
#                     else:
#                         multi_line = True
#                     mini_list = []
#                     for i in range(len(l_split)-1):
#                         mini_list.append(float(l_split[i]))
#                     if len(mini_list) == 0:
#                         pass
#                     else:
#                         new_list.append(mini_list)
#                         
#     if len(new_list) == 1:
#         interaction_parameters[(type0, type1)][new_parameter] = new_list[0]
#     elif len(new_list) > 1:
#         interaction_parameters[(type0, type1)][new_parameter] = new_list
#                         
#     for key, val in interaction_parameters.items():
#         for k, v in interaction_parameters[key].items():
#             if k == 'interact':
#                 for i in range(len(interaction_parameters[key][k])):
#                     if type(interaction_parameters[key][k]) == list:
#                         try:                            
#                             for j in range(len(interaction_parameters[key][k][i])):
#                                 interaction_parameters[key][k][i][j] = int(interaction_parameters[key][k][i][j])
#                         except:
#                             interaction_parameters[key][k][i] = int(interaction_parameters[key][k][i])
#                     else:
#                         interaction_parameters[key][k] = int(interaction_parameters[key][k])
#             else:
#                 pass
#                         
#     return interaction_parameters
# =============================================================================


def import_interaction_params(filename):
    multiline_params = {"interact", "p_ass", "p_diss"}
    interaction_parameters = {}
    with open(filename, "r") as f:
        lines = [line.strip() for line in f]
    current_pair = None
    current_param = None
    
    i = 0
    while i < len(lines):
        line = lines[i]
        # Blank line means next pairwise interaction block
        if line == "":
            i += 1
            continue
        current_param = line
        i += 1
        if current_param == "type":
            tokens = [int(float(x)) for x in lines[i].split(",")[:-1]]
            current_pair = tuple(tokens)
            interaction_parameters[current_pair] = {}
            i += 1
            continue
        if current_param in multiline_params:
            values = []
            while i < len(lines):
                l = lines[i]
                if l == "":
                    break
                if "," not in l:
                    break
                tokens = l.split(",")[:-1]
                if current_param == "interact":
                    row = [int(float(x)) for x in tokens]
                else:
                    row = [float(x) for x in tokens]
                values.append(row)
                i += 1
            interaction_parameters[current_pair][current_param] = values
            continue
    return interaction_parameters


def import_simulation_params(filename):
    simulation_params = {}
    new_parameter = 'dummy'
    multi_line = False
    start_new = False
    particle_number = 0
    simulation_params[str(particle_number)] = {}
    new_list = []
    int_flag = False
    with open(filename, 'r') as file:
        for line in file:
            l = line.strip()
            l_split = l.split(',')

            if len(l_split) <= 2:
                if len(new_list) == 0:
                    pass
                elif len(new_list) == 1:
                    simulation_params[str(particle_number)][new_parameter] = new_list[0]
                else:
                    simulation_params[str(particle_number)][new_parameter] = new_list
                multi_line = False
                new_list = []
                if l_split[0] == '':
                    start_new = True
                    particle_number += 1
                # Prepare next parameter entry
                else:
                    if start_new == True:
                        simulation_params[str(particle_number)] = {}
                        start_new = False
                    else:
                        pass
                    if l_split[0].lower() == 'yes' or l_split[0].lower() == 'true':
                        simulation_params[str(particle_number)]['interacting'] = True
                    elif l_split[0].lower() == 'no' or l_split[0].lower() == 'false':
                        simulation_params[str(particle_number)]['interacting'] = False
                    else:
                        pass
                    try:
                        simulation_params[str(particle_number)][new_parameter] = float(l_split[0])
                    except:
                        new_parameter = l_split[0]
            else:
                multi_line = True
                mini_list = []
                for i in range(len(l_split)-1):
                    mini_list.append(float(l_split[i]))
                new_list.append(mini_list)

    for key, val in simulation_params.items():
        for k, v in simulation_params[key].items():
            if k not in ['box_limits', 'viscosity', 'temperature']:
                simulation_params[key][k] = int(v)
            else:
                pass
            
    simulation_parameters = simulation_params['0']
    
    return simulation_parameters
                    
def import_patches(file):
    inter = []
    position = []
    alpha = []
    prob_ass = []
    prob_diss = []
    with open(file) as f:
        which = ''
        for line in f:
            row = line.strip().split(",")
            if len(row) == 1:
                if len(row[0]) == 0:
                    pass
                else:
                    which = row[0]
            else:
                if which == 'position':
                    entry = []
                    for i in range(len(row)):
                        entry.append(float(row[i]))
                    position.append(entry)
                elif which == 'alpha':
                    alpha.append(float(row[0]))
                elif which == 'interactions':
                    entry = []
                    for i in range(len(row)):
                        entry.append(int(row[i]))
                    inter.append(entry)
                elif which == 'association':
                    entry = []
                    for i in range(len(row)):
                        entry.append(float(row[i]))
                    prob_ass.append(entry)
                elif which == 'dissociation':
                    entry = []
                    for i in range(len(row)):
                        entry.append(float(row[i]))
                    prob_diss.append(entry)
                else:
                    print('Error! Double check input file!')
                    SystemExit()
    position_norm = np.array(position) / np.linalg.norm(position)
    return position_norm, np.array(alpha), np.array(inter), \
           np.array(prob_ass), np.array(prob_diss)
         
# Group particle parameters related to the pathes in a nested dictionary 'patches'
def group_patch_params(parameters):
    keys_to_delete = []
    for keys, values in parameters.items():
        parameters[keys]['patches'] = {}
        for key, val in values.items():
            if 'patches_' in key:
                key_split = key.split('_')
                if key not in keys_to_delete:
                    keys_to_delete.append(key)
                parameters[keys]['patches'][key_split[1]] = val
            else:
                pass
    return parameters
           
def get_dict_params(dictionary, target_key):
    for key, val in dictionary.items():
        if target_key == key:
            pars = val
        else:
            pass
    return pars
           
def get_interaction_params(inter_params, type1, type2):
    if type1 > type2:
        aux = type1
        type1 = type2
        type2 = aux
    else:
        pass
    params = inter_params[(type1, type2)]
    return params

def copy_dict(entry_dict):
    new_dict = {}
    for key, val in entry_dict.items():
        new_dict[key] = val
    return new_dict

@njit
def update_cell_list(coords, box_limits, cell_size, n_cells_xyz):
    total_cells = n_cells_xyz[0] * n_cells_xyz[1] * n_cells_xyz[2]
    head = np.full(total_cells, -1, dtype=np.int32)
    linked_list = np.full(len(coords), -1, dtype=np.int32)
    for i in range(len(coords)):
        # Find cell indices (0 to n_cells-1)
        ix = int((coords[i, 0] + box_limits) / cell_size)
        iy = int((coords[i, 1] + box_limits) / cell_size)
        iz = int((coords[i, 2] + box_limits) / cell_size)
        # Clip to ensure floating point errors don't go out of bounds
        ix = max(0, min(ix, n_cells_xyz[0]-1))
        iy = max(0, min(iy, n_cells_xyz[1]-1))
        iz = max(0, min(iz, n_cells_xyz[2]-1))
        c_idx = ix + n_cells_xyz[0] * (iy + n_cells_xyz[1] * iz)
        linked_list[i] = head[c_idx]
        head[c_idx] = i
    return head, linked_list

@njit
def check_steric_clash_cell(i_idx, i_new_pos, all_coords, head, linked_list, radii, box_limits, cell_size, n_cells_xyz, skip_indices = None):
    # Cell where the new coordinate would be
    ix = int((i_new_pos[0] + box_limits) / cell_size)
    iy = int((i_new_pos[1] + box_limits) / cell_size)
    iz = int((i_new_pos[2] + box_limits) / cell_size)
    # Check neighboring cells
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            for dz in range(-1, 2):
                nx = (ix + dx) % n_cells_xyz[0]
                ny = (iy + dy) % n_cells_xyz[1]
                nz = (iz + dz) % n_cells_xyz[2]
                target_cell = nx + n_cells_xyz[0] * (ny + n_cells_xyz[1] * nz)
                j = head[target_cell]
                while j != -1:
                    in_cluster = False
                    for ind in skip_indices:
                        if j == ind:
                            in_cluster = True
                            break
                    
                    if in_cluster == False:
                        dist_sq = calc.calculate_distance_sq(box_limits, i_new_pos, all_coords[j])
                        limit = radii[i_idx] + radii[j]
                        if dist_sq < limit**2:
                            return True # Steric clash
                    j = linked_list[j]
    return False # No clash
    
def check_particle_interaction(inter_params, particle1, particle2):
    pair = (particle1, particle2)
    interact = False
    for key in inter_params.keys():
        if pair == key:
            interact = True
        else:
            pass
    return interact

def check_patch_interaction(interaction_params_type, patch1, patch2):
    matrix = interaction_params_type['interact']
    if matrix[patch1][patch2] == 1:
        return True
    else:
        return False
    
def check_distance(r, distance):
    if (distance < r):
        return True
    else:
        return False

def max_distance(part_params, manual_value = 0):
    if manual_value != 0:
        return manual_value
    else:
        max_hs = 0
        max_patch = 0
        for key, val in part_params.items():
            for k, v in part_params[key].items():
                if k == 'radius':
                    hs_candidate = part_params[key][k]
                    if hs_candidate > max_hs:
                        max_hs = hs_candidate
                    else:
                        pass
                elif k == 'patches':
                    for kk, vv in part_params[key][k].items():
                        if kk == 'radius':
                            patch_candidates = part_params[key][k][kk]
                            try: # If more than one patch (patch_candidates is a list)
                                for i in range(len(patch_candidates)):
                                    if patch_candidates[i] > max_patch:
                                        max_patch = patch_candidates[i]
                                    else:
                                        pass
                            except: # If only one patch (patch_candidates is a float)
                                if patch_candidates > max_patch:
                                    max_patch = patch_candidates
                                else:
                                    pass                                
                        else:
                            pass
                else:
                    pass
    maximum_distance = 2*max_hs + 2*max_patch
    return maximum_distance

# Given two patches in n_norm and a d_norm as the distance vector between both particles, checks if they are aligned
@njit
def check_alignment(n_norms, d_norm, alphas, degrees = True):
    alphas_radians = np.zeros(len(alphas), dtype=np.float64)
    if degrees == True:
        for i in range(len(alphas)):
            alphas_radians[i] = alphas[i] * (np.pi / 180.0)
    else:
        for i in range(len(alphas)):
            alphas_radians[i] = alphas[i]
    aligned = np.array([False, False])
    sign = -1.0
    for i in range(len(alphas_radians)):
        dot = np.dot(n_norms[i, :], sign*d_norm)
        if dot >= np.cos(alphas_radians[i]):
            aligned[i] = True
        else:
            pass
        sign = -sign
    if aligned[0] == True and aligned[1] == True:
        return True
    else:
        return False

# Given two lists of patches, checks which are aligned
def check_alignment_all(patches1, patches2, alphas1, alphas2, d_vector):
    d_norm = np.ascontiguousarray(d_vector / np.linalg.norm(d_vector))
    aligned_pair = []
    check = False
    for i in range(len(patches1)):
        patch1 = patches1[i]
        a1 = alphas1[i]
        for j in range(len(patches2)):
            patch2 = patches2[j]
            a2 = alphas2[j]
            n_norms = np.zeros((2, 3), dtype=np.float64)
            n_norms[0, :] = patch1
            n_norms[1, :] = patch2
            alphas = np.array([a1, a2], dtype=np.float64)
            check_alignment_patches = check_alignment(n_norms, d_norm, alphas)
            if check_alignment_patches == True:
                check = True
                aligned_pair = [i,j]
                return True, [i,j]
                #return check, aligned_pair
    return check, aligned_pair

def count_all_patches(sim_params, patches):
    n_dims = sim_params['n_dimensions']
    count = 0
    for p in patches:
        if isinstance(p, (list, np.ndarray)):
            for x in p:
                if isinstance(x, (list, tuple, np.ndarray)) and len(x) == n_dims:
                    count += 1
    return count

# Extract cluster sizes and maximum population
def obtain_cluster_features(sim_params, clusters):
    cluster_number = len(clusters)
    if cluster_number == 0:
        cluster_agg = [0]
        n_max = 0
        return cluster_number, cluster_agg, n_max
    cluster_agg = []
    for i in range(len(clusters)):
        cluster_agg.append(len(clusters[i]))
    n_max = np.max(np.array(cluster_agg))
    return cluster_number, cluster_agg, n_max

# Find number of bonds per patch from interaction matrix
def find_bonds(part_params, simulation, particle_index):
    n_patches = len(simulation['patches'][particle_index])
    bonds = [0 for i in range(n_patches)]
    interactions = simulation['interactions']
    for i in range(len(interactions)):
        patch = int(interactions[particle_index][i][0])
        if patch >= 0:
            #print(interactions[particle_index][i])
            #print(bonds)
            #print(f"Particle {particle_index} (type {simulation['particles'][particle_index]}), Bonds: {len(bonds)}, Patch index: {patch}")
            bonds[patch] += 1
    return bonds
        
# Finds bonds for all particles from interacion matrix
def find_bonds_all(part_params, simulation):
    bonds_all = []
    for idx, val in enumerate(simulation['particles']):
        bonds_part = find_bonds(part_params, simulation, idx)
        bonds_all.append(bonds_part)
    return bonds_all

def test_association(patch1, patch2, prob):
    random = np.random.uniform(low = 0, high = 1)
    if random < prob:
        inter_element = [patch1,patch2]
        #if patch1 > patch2:
        #    inter_element = [patch2,patch1]
        #else:
        #    inter_element = [patch1,patch2]
    else:
        inter_element = [-1,-1]
    return inter_element

def test_dissociation(patch1, patch2, prob):
    random = np.random.uniform(low = 0, high = 1)
    if random < prob:
        inter_element = [-1,-1]
    else:
        inter_element = [patch1,patch2]
        #if patch1 > patch2:
        #    inter_element = [patch2,patch1]
        #else:
        #    inter_element = [patch1,patch2]
    return inter_element


# -------------------------------------------------
# ------CALCULATE PROBABILITIES FROM ENERGIES------
# -------------------------------------------------

# For each ENTRY of the dictionary generated from importing an interacions_params file,
# calculate p_ass and p_diss from energy matrix if not explicitly given in the file
def add_probs_from_energies(inter_params_entry):
    tol = 1e-8
    keys = list(inter_params_entry.keys())
    if 'p_ass' not in keys:
        p_ass = np.zeros_like(inter_params_entry['energies'])
        for r, energies_row in enumerate(inter_params_entry['energies']):
            for i, energy_val in enumerate(energies_row):
                p_ass[r,i] = calc.calculate_probability_from_energy(energy_val)
                if p_ass[r,i] < tol or inter_params_entry['energies'][r][i] == 0:
                    p_ass[r,i] = 0
                elif p_ass[r,i] > 1.0 - tol:
                    p_ass[r,i] = 1
        inter_params_entry['p_ass'] = p_ass
    
    if 'p_diss' not in keys:
        p_diss = np.zeros_like(inter_params_entry['energies'])
        for r, energies_row in enumerate(inter_params_entry['energies']):
            for i, energy_val in enumerate(energies_row):
                p_diss[r,i] = calc.calculate_probability_from_energy(-energy_val)
                if p_diss[r,i] < tol:
                    p_diss[r,i] = 0
                elif p_diss[r,i] > 1.0 - tol:
                    p_diss[r,i] = 1
        inter_params_entry['p_diss'] = p_diss
            


# ------------------------------
# ------EXPORT FILES BLOCK------
# ------------------------------

# Write .tcl file for VMD visualization
def write_vmd_config(path, sim_params, part_params, simulation):
    atoms_com = ['C', 'N', 'O', 'F', 'Ne']
    atoms_patches = ['H', 'He', 'Li', 'Be', 'B']
    n_parts = len(simulation['coordinates'])
    n_patches = count_all_patches(sim_params, simulation['patches'])
    n_total = n_parts + n_patches
    with open(path, 'w') as fp:
        fp.write('package require pbctools\n')
        fp.write('\n')
        fp.write('mol modstyle 0 top VDW\n')
        fp.write('foreach {elem rad} {\n')
        # LOOP
        fp.write(f'    ')
    return 0

# Change patch per particle to not repeat the same element for different particles
def write_coords_xyz(path, sim_params, part_params, simulation, frame_number = 0, new_file = False):
    atoms_com = ['C', 'N', 'O', 'F', 'Ne', 'Al', 'Si', 'P', 'S', 'Cl']
    atoms_patches = ['H', 'He', 'Li', 'Be', 'B', 'Na', 'Mg', 'K', 'Ca', 'Sc']
    n_parts = len(simulation['coordinates'])
    n_patches = count_all_patches(sim_params, simulation['patches'])
    n_total = n_parts + n_patches
    if new_file == True:
        mode = 'w'
    else:
        mode = 'a'
    with open(path, mode) as fp:
        fp.write(f'{n_total}\n')
        fp.write(f'Trajectory: Frame {frame_number}/{sim_params['n_steps']}\n')
        for i in range(n_parts):
            part_type = simulation['particles'][i]
            part_radius = part_params[str(part_type)]['radius']
            coords = simulation['coordinates'][i]
            fp.write(f"{atoms_com[part_type]}    {coords[0]}    {coords[1]}    {coords[2]}\n")
            patches_i = simulation['patches'][i]
            # Change here!
            patch_index = 0
            for x in patches_i:
                if isinstance(x, (list, tuple, np.ndarray)) and len(x) == sim_params['n_dimensions']:
                    patches_coords = (x * part_radius) + coords
                    atom_label = atoms_patches[patch_index % len(atoms_patches)]
                    fp.write(f"{atom_label}    {patches_coords[0]}    {patches_coords[1]}    {patches_coords[2]}\n")
                    patch_index += 1
                    
def write_coords_csv(path, sim_params, part_params, simulation, frame_number = 0, new_file = False):
    n_parts = len(simulation['coordinates'])
    n_patches = count_all_patches(sim_params, simulation['patches'])
    n_total = n_parts + n_patches
    if new_file == True:
        mode = 'w'
    else:
        mode = 'a'
    with open(path, mode) as fp:
        fp.write(f'{frame_number},{sim_params['n_steps']}\n')
        for i in range(n_parts):
            coords = simulation['coordinates'][i]
            fp.write(f"{coords[0]},{coords[1]},{coords[2]}\n")
            patches_i = simulation['patches'][i]
            patch_index = 0

def write_matrix(path, sim_params, matrix, frame_number = 0, new_file = False):
    if new_file == False:
        mode = 'a'
    else:
        mode = 'w'
    with open(path, mode) as fp:
        fp.write(f'Frame {frame_number}/{sim_params['n_steps']}\n')
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                fp.write(f'{matrix[i][j]},')
            fp.write('\n')
            
def write_clusters(path, clusters, frame_number = 0, new_file = False):
    if new_file == False:
        mode = 'a'
    else:
        mode = 'w'
    with open(path, mode) as fp:
        fp.write(f'Frame {frame_number}\n')
        for idx, cluster in enumerate(clusters):
            for i, val in enumerate(cluster):
                if i == (len(cluster)-1):
                    fp.write(f'{val}\n')
                else:
                    fp.write(f'{val},')
        fp.write('\n')
            
def write_clusters_feats(path, cluster_feats, frame_number = 0, new_file = False):
    #print(cluster_feats)
    if new_file == False:
        mode = 'a'
    else:
        mode = 'w'
    with open(path, mode) as fp:
        for i in range(len(cluster_feats)):
            if i == len(cluster_feats):
                fp.write(f'{cluster_feats[i]}')
            else:
                fp.write(f'{cluster_feats[i]},')
        fp.write('\n')
        
def write_clusters_int(path, cluster_int, frame_number = 0, new_file = False):
    #print(cluster_max)
    if new_file == False:
        mode = 'a'
    else:
        mode = 'w'
    with open(path, mode) as fp:
        fp.write(f'{cluster_int}')
        fp.write('\n')
            
def write_particle_types(path, simulation):
    with open(path, 'w') as fp:
        for i in range(len(simulation['particles'])):
            fp.write(f"{simulation['particles'][i]}\n")
            
            
# ------------------------------
# ------EXPORT FILES BLOCK------
# ------------------------------

# Converts particle indices into cluster indices
def particle_cluster_indexing(simulation, clusters_prev):
    which_cluster_prev = -1 * np.ones(len(simulation['particles']))
    for c, cluster in enumerate(clusters_prev):
        for idx, part in enumerate(cluster):
            which_cluster_prev[part] = c
    return which_cluster_prev

# Creates list of clusters labeled only with the cluster index (e.g., [[0,0,0], [1,1], [2,2,2,2,2], ...])
def clusters_origins_reference(clusters_prev):
    reference_clusters = []
    for c, cluster in enumerate(clusters_prev):
        origin = []
        for p in range(len(cluster)):
            origin.append(c)
        reference_clusters.append(origin)
    return reference_clusters

# Creates list with new clusters indicating from which cluster the particles came from (e.g, [0,0,1], [1,-1], [2,2,2,2,2], ...])
def find_cluster_origins(simulation, clusters_prev):
    which_cluster_prev = particle_cluster_indexing(simulation, clusters_prev)
    part_indices = range(len(simulation['particles']))
    cluster_ids_from_prev = [cluster.copy() for cluster in simulation['clusters']]
    # Sets every element as -1 to "forget" clusters
    for c in range(len(cluster_ids_from_prev)):
        for i in range(len(cluster_ids_from_prev[c])):
            cluster_ids_from_prev[c][i] = -1
    # Iterates through clusters
    for idx in part_indices:
        for c, cluster in enumerate(simulation['clusters']):
            for i, cl in enumerate(cluster):
                if idx == cl:
                    cluster_ids_from_prev[c][i] = int(which_cluster_prev[idx])
    return cluster_ids_from_prev

# Assitant function for calculating the transitions matrix
def count_dissociated_monomers(previous, current):
    counts = []
    for c, clust in enumerate(previous):
        total = len(clust)
        counter = 0
        for i, curr in enumerate(current):
            unique = np.unique(curr, return_counts=True)
            types = unique[0]
            cnts = unique[1]
            for j in range(len(types)):
                if c == types[j]:
                    counter += cnts[j]
        counts.append(int(total - counter))
    return counts

# Builds the transitions matrix (previous clusters are read on rows and current on columns). Extra row and column for monomers
def generate_transitions_matrix(previous, current):
    matrix_len = np.max([len(previous), len(current)]) + 1
    tracking_matrix = np.zeros((matrix_len, matrix_len), dtype = int)
    for c, clust in enumerate(current):
        unique = np.unique(clust, return_counts=True)
        origins = unique[0]
        counts = unique[1]
        for o in range(len(origins)):
            tracking_matrix[origins[o],c] = int(counts[o])
    dissociated_monomers = count_dissociated_monomers(previous, current)
    for m, mon in enumerate(dissociated_monomers):
        tracking_matrix[m,-1] = int(mon)
    return tracking_matrix

# Finds transitions from row (dissociations)
def assign_transitions_row(row):
    transitions = []
    non_zero_idx = np.nonzero(row)[0]
    if len(non_zero_idx) > 1:
        non_zero_pops = row[non_zero_idx]
        prev_agg = np.sum(non_zero_pops)
        passed = False
        for i in range(len(non_zero_idx)):
            monomers_idx = len(row)-1
            monomers = row[monomers_idx]
            if passed == True:
                if i == monomers_idx:
                    while monomers > 0:
                        this_trans = [[int(prev_agg)], [int(prev_agg-1),1]]
                        prev_agg -= 1
                        monomers -= 1
                        transitions.append(this_trans)
                else:
                    dissociated = non_zero_pops[i]
                    this_trans = [[int(prev_agg)], [int(prev_agg-dissociated), int(dissociated)]]
                    prev_agg -= dissociated
                    transitions.append(this_trans)
            else:
                passed = True
    # Special case: total dissociation of cluster into monomers
    elif (len(non_zero_idx)) == 1 and (non_zero_idx[0] == len(row)-1):
        monomers = row[non_zero_idx[0]]
        while monomers > 1:
            this_trans = [[int(monomers)], [int(monomers-1),1]]
            monomers -= 1
            transitions.append(this_trans)
    return transitions

# Finds transitions from column (associations)
def assign_transitions_column(column):
    transitions = []
    non_zero_idx = np.nonzero(column)[0]
    if len(non_zero_idx) > 1:
        non_zero_pops = column[non_zero_idx]
        passed = False
        curr_agg = 0
        for i in range(len(non_zero_idx)):
            monomers_idx = len(column)-1
            monomers = column[monomers_idx]
            if passed == True:
                if i == monomers_idx:
                    while monomers > 0:
                        this_trans = [[int(curr_agg),1],[int(curr_agg+1)]]
                        curr_agg += 1
                        monomers -= 1
                        transitions.append(this_trans)
                else:
                    associated = non_zero_pops[i]
                    this_trans = [[int(curr_agg), int(associated)], [int(curr_agg+associated)]]
                    curr_agg += associated
                    transitions.append(this_trans)
            else:
                curr_agg = non_zero_pops[i]
                passed = True
    # Special case: formation of new cluster from monomers
    elif (len(non_zero_idx)) == 1 and (non_zero_idx[0] == len(column)-1):
        monomers = column[non_zero_idx[0]]
        counter = 0
        while monomers > 1:
            this_trans = [[counter + 1, 1], [counter + 2]]
            monomers -= 1
            counter += 1
            transitions.append(this_trans)
    return transitions

# Find all transitions from transitions matrix 
# (output is a list (all) of list (transitions from/to specific cluster) of lists (each transition))
def assign_transitions(transitions_matrix):
    transitions = []
    for r, row in enumerate(transitions_matrix):
        limit = len(transitions_matrix)-1
        if r < limit:
            trans = assign_transitions_row(row)
            for t in trans:
                transitions.append(trans)
    for c, col in enumerate(transitions_matrix.T):
        limit = len(transitions_matrix.T)-1
        if c < limit:
            trans = assign_transitions_column(col)
            for t in trans:
                transitions.append(trans)
    return transitions

# Exports transitions to .csv file
def write_transitions(path, transitions, frame_number = 0, new_file = False):
    if new_file == False:
        mode = 'a'
    else:
        mode = 'w'
    with open(path, mode) as fp:
        for trans_clust in transitions:
            for trans in trans_clust:
                fp.write(f'{frame_number},')
                for i, t0 in enumerate(trans[0]):
                    if i < len(trans[0])-1:
                        fp.write(f'{t0},')
                    else:
                        fp.write(f'{t0}')
                fp.write(',-,')
                for j, t1 in enumerate(trans[1]):
                    if j < len(trans[1])-1:
                        fp.write(f'{t1},')
                    else:
                        fp.write(f'{t1}')
                fp.write('\n')
                
# Import transitions from .csv file to two lists: one for associations and another for dissociations
# Entries: association -> [frame_number, [a,b], a+b] or dissociation -> [frame_number, a+b, [a,b]]
def import_transitions(path):
    associations = []
    dissociations = []
    with open(path, 'r') as fp:
        counter = 1
        for line in fp:
            split = line.split(',')
            if split[2] == '-':
                dissociations.append([int(split[0]), [int(split[1])], [int(split[3]), int(split[4])]])
            elif split[3] == '-':
                associations.append([int(split[0]), [int(split[1]), int(split[2])], [int(split[4])]])
            else:
                print(f'Strange line at line {counter}')
            counter += 1
    return associations, dissociations