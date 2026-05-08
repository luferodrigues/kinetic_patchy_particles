import numpy as np
from scipy.spatial.transform import Rotation as R
import utils

# Simple distance calculation
def calculate_distance_simple(vec1, vec2):
    vector = vec2 - vec1
    dist = np.linalg.norm(vector)
    return dist

# Cartesian distance definition considering periodic boundary conditions from -box_limits to +box_limits
def calculate_distance(sim_params, vec1, vec2):
    dist_squared = 0
    for i in range(len(vec1)):
        diff1 = (vec2[i] - vec1[i])**2
        diff2 = (vec2[i] - (vec1[i] + 2*sim_params['box_limits']))**2
        diff3 = (vec2[i] - (vec1[i] - 2*sim_params['box_limits']))**2
        dist_squared += min([diff1, diff2, diff3])
    dist = np.sqrt(dist_squared)
    return dist

# Calculates center of mass of list of coordinates
def center_of_mass(coordinates_list):
    com = np.array([np.sum(coordinates_list[:,0]), np.sum(coordinates_list[:,1]), \
                               np.sum(coordinates_list[:,2])]) / len(coordinates_list)
    return com

# Sign function
def sign(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 1 # Might also return 0
    
def normalize_vector(vector):
    return vector / np.linalg.norm(vector)

# Convert from spherical to cartesian coordinate system (r, theta E [0,pi], phi E [0,2pi))
def spherical_to_cartesian(r, theta, phi, degrees = False):
    if degrees == True:
        theta_val = theta * (np.pi / 180)
        phi_val = phi * (np.pi / 180)
    else:
        theta_val = theta
        phi_val = phi
    x = r * np.sin(theta_val) * np.cos(phi_val)
    y = r * np.sin(theta_val) * np.sin(phi_val)
    z = r * np.cos(theta_val)
    return [x, y, z]

# Convert from cartesian to spherical coordinate system (r, theta E [0,pi], phi E [0,2pi))
def cartesian_to_spherical(x, y, z, degrees = False):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)
    phi = sign(y) * np.arccos(x / (np.sqrt(x**2 + y**2)))
    #theta = np.arctan2(y, x)
    #phi = np.arctan2(z, np.sqrt(x**2 + y**2))
    if degrees == True:
        theta_val = theta * (180 / np.pi)
        phi_val = phi * (180 / np.pi)
    else:
        theta_val = theta
        phi_val = phi
    return [r, theta_val, phi_val]

# Apply rotation to a vector centered at "center" (default: origin)
def rotate_vector(vector, rotation_vector):
    rot = R.from_rotvec(rotation_vector)
    rotated = rot.apply(vector)
    return rotated

# Generate rotation object for batch use
def generate_rotation(rotation_vector):
    rot = R.from_rotvec(rotation_vector)
    return rot

# Calculate translational diffusion coeficient from hard sphere radius, viscosity and temperature
# in nm, Pa*s, // and K, respectively. Final answer in nm^2/ns
def calculate_diff_trans(radius, viscosity = 0.8539e-3, temperature = 300):
    if (viscosity == 0):
        raise ValueError('viscosity cannot be zero!')
    if (radius == 0):
        raise ValueError('radius cannot be zero!')
    k_B = 1.380649 * 1e-23 # in J/K
    diffusion_trans = k_B * temperature / (6 * np.pi * viscosity * 1e-9 * radius)
    return diffusion_trans * 1e9 # Converting from m^2/s to to nm^2/ns

# Calculate rotational diffusion coeficient from hard sphere radius, viscosity and temperature
# in nm, Pa*s, // and K, respectively. Final answer in rad^2/ns
def calculate_diff_rotation(radius, viscosity = 0.8539e-3, temperature = 300):
    if (viscosity == 0):
        raise ValueError('viscosity cannot be zero!')
    if (radius == 0):
        raise ValueError('radius cannot be zero!')
    k_B = 1.380649 * 1e-23 # in J/K
    diff_rot = k_B * temperature / (8 * np.pi * viscosity * (1e-9 * radius)**3)
    return diff_rot * 1e9 # # Converting from rad^2/s to to rad^2/ns

# Converts rotational/rotational diffusion into angle per unit time
def diff_to_sigma(diff, time_step):
    sigma = np.sqrt(2*diff*time_step)
    return sigma

# Estimate diffusion coefficient from cluster
def diffusion_cluster(part_params, simulation, process_list, which = 'trans'):
    n = len(process_list)
    diff = 1e14
    for p in process_list:
        part_type = simulation['particles'][p]
        if which == 'trans':
            diff_handle = part_params[str(int(part_type))]['diff_trans']
        elif which == 'rot':
            diff_handle = part_params[str(int(part_type))]['diff_rot']
        else:
            SystemExit()
        if diff_handle < diff:
            diff = diff_handle
        else:
            pass
    diff_cluster = diff / np.sqrt(n)
    return diff_cluster

def calculate_gr(parameters, dist_matrix, n_bins = 200):
    d = []
    histogram = np.histogram(dist_matrix, bins = n_bins)
    hist = utils.correct_histogram(histogram)
    r = hist[0]
    d.append(hist[1])
    d = np.array(d)
    gr = np.zeros(len(r))
    n = 0
    v = 2*parameters['box_limits']**3
    average_density = n / v
    n = parameters['n_particles'] * parameters['fraction_interacting'] * 4 * np.pi
    for i in range(1, len(r)):
        n_r = d[:,1]
        r_term = r[i]**3 - r[i-1]**3
        #r_term = r[i]**2 * delta_r # if delta_r is small enough
        for j in range(len(n_r)):
            gr[i] += n_r[j] / r_term
    return [r, v / n * gr]

# Calculate S(q) from g(r) (REF: Cristiano thesis, p. 213)
def calculate_sq(parameters, q, r, gr):
    v = parameters['box_limits']**3
    n = parameters['n_particles'] * parameters['fraction_interacting' * 4]
    integral = np.zeros(len(q))
    for i in range(len(q)):
        integrand = r * np.sin(q*r) / q * (gr - 1)
        integral[i] = np.trapezoid(integrand, x = r)
    return 1 + n/v * 4*np.pi * integral
