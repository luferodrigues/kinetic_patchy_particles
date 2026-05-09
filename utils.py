import os
import numpy as np

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

def import_particle_params(filename):
    particle_parameters = {}
    new_parameter = 'dummy'
    multi_line = False
    start_new = False
    particle_number = 0
    particle_parameters[str(particle_number)] = {}
    new_list = []
    with open(filename, 'r') as file:
        for line in file:
            l = line.strip()
            l_split = l.split(',')

            if len(l_split) <= 2:
                if len(new_list) == 0:
                    pass
                elif len(new_list) == 1:
                    particle_parameters[str(particle_number)][new_parameter] = new_list[0]
                else:
                    particle_parameters[str(particle_number)][new_parameter] = new_list
                multi_line = False
                new_list = []
                if l_split[0] == '':
                    start_new = True
                    particle_number += 1
                # Prepare next parameter entry
                else:
                    if start_new == True:
                        particle_parameters[str(particle_number)] = {}
                        start_new = False
                    else:
                        pass
                    if l_split[0].lower() == 'yes' or l_split[0].lower() == 'true':
                        particle_parameters[str(particle_number)]['interacting'] = True
                    elif l_split[0].lower() == 'no' or l_split[0].lower() == 'false':
                        particle_parameters[str(particle_number)]['interacting'] = False
                    try:
                        particle_parameters[str(particle_number)][new_parameter] = float(l_split[0])
                    except:
                        new_parameter = l_split[0]
            else:
                multi_line = True
                mini_list = []
                for i in range(len(l_split)-1):
                    mini_list.append(float(l_split[i]))
                new_list.append(mini_list)

    for key, val in particle_parameters.items():
        for k, v in particle_parameters[key].items():
            if k == 'type' or k == 'number':
                particle_parameters[key][k] = int(v)
            else:
                pass
    
    grouped_particle_parameters = group_patch_params(particle_parameters)
    return grouped_particle_parameters

def import_interaction_params(filename):
    interaction_parameters = {}
    new_parameter = 'dummy'
    multi_line = False
    start_new = False
    new_interaction = False
    new_list = []
    type0 = 0
    type1 = 0
    with open(filename, 'r') as file:
        for line in file:
            l = line.strip()
            l_split = l.split(',')
            
            if l_split[0] == 'type':
                new_interaction = True
            else:
                if new_interaction == True:
                    type0 = int(float(l_split[0]))
                    type1 = int(float(l_split[1]))
                    interaction_parameters[(type0, type1)] = {}
                    new_interaction = False
                else:
                    if len(l_split) <= 2:
                        if len(new_list) == 0:
                            pass
                        elif len(new_list) == 1:
                            interaction_parameters[(type0, type1)][new_parameter] = new_list[0]
                        else:
                            interaction_parameters[(type0, type1)][new_parameter] = new_list
                        if len(l_split) == 1:
                            new_parameter = l_split[0]
                        else:
                            pass
                        multi_line = False
                        new_list = []
                    else:
                        multi_line = True
                    mini_list = []
                    for i in range(len(l_split)-1):
                        mini_list.append(float(l_split[i]))
                    if len(mini_list) == 0:
                        pass
                    else:
                        new_list.append(mini_list)
                        
    if len(new_list) == 1:
        interaction_parameters[(type0, type1)][new_parameter] = new_list[0]
    elif len(new_list) > 1:
        interaction_parameters[(type0, type1)][new_parameter] = new_list
                        
    for key, val in interaction_parameters.items():
        for k, v in interaction_parameters[key].items():
            if k == 'interact':
                for i in range(len(interaction_parameters[key][k])):
                    for j in range(len(interaction_parameters[key][k][i])):
                        interaction_parameters[key][k][i][j] = int(interaction_parameters[key][k][i][j])
            else:
                pass
                        
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
                            for i in range(len(patch_candidates)):
                                if patch_candidates[i] > max_patch:
                                    max_patch = patch_candidates[i]
                                else:
                                    pass
                        else:
                            pass
                else:
                    pass
    maximum_distance = 2*max_hs + 2*max_patch
    return maximum_distance

# Given two patches in n_norm and a d_norm as the distance vector between both particles, checks if they are aligned
def check_alignment(n_norms, d_norm, alphas, degrees = True):
    alphas_radians = np.zeros_like(alphas, dtype = float)
    if degrees == True:
        for i in range(len(alphas)):
            alphas_radians[i] = alphas[i] * (np.pi / 180)
    else:
        alphas_radians = alphas
    aligned = [False, False]
    sign = -1
    for i in range(len(alphas_radians)):
        dot = np.dot(n_norms[i], sign*d_norm)
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
    d_norm = d_vector / np.linalg.norm(d_vector)
    aligned_pair = []
    check = False
    for i in range(len(patches1)):
        patch1 = patches1[i]
        a1 = alphas1[i]
        for j in range(len(patches2)):
            patch2 = patches2[j]
            a2 = alphas2[j]
            n_norms = [patch1, patch2]
            alphas = [a1, a2]
            check_alignment_patches = check_alignment(n_norms, d_norm, alphas)
            if check_alignment_patches == True:
                check = True
                aligned_pair = [i,j]
                return True, [i,j]
                #return check, aligned_pair
    return check, aligned_pair

def test_association(patch1, patch2, prob):
    random = np.random.uniform(low = 0, high = 1)
    if random < prob:
        if patch1 > patch2:
            inter_element = [patch2,patch1]
        else:
            inter_element = [patch1,patch2]
    else:
        inter_element = [-1,-1]
    return inter_element

def test_dissociation(patch1, patch2, prob):
    random = np.random.uniform(low = 0, high = 1)
    if random < prob:
        inter_element = [-1,-1]
    else:
        if patch1 > patch2:
            inter_element = [patch2,patch1]
        else:
            inter_element = [patch1,patch2]
    return inter_element

# ------------------------------
# ------EXPORT FILES BLOCK------
# ------------------------------

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

def write_coords_xyz(path, sim_params, part_params, simulation, frame_number = 0, new_file = False):
    atoms_com = ['C', 'N', 'O', 'F', 'Ne']
    atoms_patches = ['H', 'He', 'Li', 'Be', 'B']
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