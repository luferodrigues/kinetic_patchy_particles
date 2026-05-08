import numpy as np
import calculate as calc
import utils

# Add total number of particles to parameters
def add_n_particles(parameters, particles_params):
    parameters['n_particles'] = utils.count_particles(particles_params)

# Generate starting coordinate values for all particles
def initialize_coordinates(sim_params, part_params):
    sim_params['n_particles'] = utils.count_particles(part_params)
    return np.random.uniform(low = -sim_params['box_limits'], high = sim_params['box_limits'], \
                             size = (sim_params['n_particles'], 3))
        
# Generate starting rotation angles (n_dimensions - 1) for all particles
def initialize_patches(sim_params, part_params, simulation):
    rotated_patches = []
    for i in range(len(simulation['particles'])):
        part_type = str(simulation['particles'][i])
        if part_params[part_type]['interacting'] == True:
            rot_vec = np.random.uniform(low = 0, high = 1, size=3)
            rot_particle = []
            for j in range(len(part_params[part_type]['patches']['positions'])):
                rot_patch = calc.rotate_vector(part_params[part_type]['patches']['positions'][j], rot_vec)
                rot_particle.append(rot_patch/np.linalg.norm(rot_patch))
        else:
            rot_particle = []
        rotated_patches.append(rot_particle)
    return rotated_patches

# Initialize distances with 1e4 or with NaN
def initialize_distances(sim_params):
    matrix = 1e4 * np.ones((sim_params['n_particles'], sim_params['n_particles']))
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if j <= i:            
                matrix[i][j] = np.nan
    return matrix

# Find the maximum number of patches per particle
def find_max_patches(particles_params):
    patches = []
    for key, val in particles_params.items():
        for k, v in particles_params[key].items():
            if k == 'patch_alphas':
                patches.append(len(particles_params[key][k]))
    max_patches = max(patches)
    return max_patches

# Initialize interaction matrix with zeros (no contact)
def initialize_interactions(sim_params, part_params):
    matrix = - np.ones((sim_params['n_particles'], sim_params['n_particles'], 2))
    return matrix

# Initialize cluster matrix with zeros (no contact)
def initialize_clusters():
    return []

# Indexes of n_particles particles according to our input particle parameters file
def initialize_particles(particles_params):
    n_total = utils.count_particles(particles_params)
    particles_list = np.zeros(n_total)
    n_types = []
    types = []
    for key, val in particles_params.items():
        for k, v in particles_params[key].items():
            if k == 'number':
                n_types.append(particles_params[key][k])
            elif k == 'type':
                types.append(particles_params[key][k])
            else:
                pass
    counter_parts = 0
    counter_types = 0
    for i in range(len(types)):
        while counter_types < n_types[i]:
            particles_list[counter_parts] = types[i]
            counter_types += 1
            counter_parts += 1
        counter_types = 0
    return np.array(particles_list, dtype = int)
