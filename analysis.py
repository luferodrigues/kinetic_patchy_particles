import numpy as np
from scipy.spatial import Delaunay

# ----------------------------------------
# ---------AUXILIARY FUNCTIONS------------
# ----------------------------------------



# ----------------------------------------
# -------IMPORT CLUSTER FEATURES----------
# ----------------------------------------

def import_cluster_n_max(file, frame_start = 1, frame_end = 100):
    feats = np.genfromtxt(file, dtype = int)
    return feats[frame_start-1:frame_end+1]
    
def import_cluster_agg(file, frame_start = 1, frame_end = 100, first_frame = 1, stride = 1):
    cluster_agg = []
    current_frame = first_frame
    with open(file) as f:
        while True:
            if current_frame > frame_end:
                break # Stop reading after end of specified frames range
            if current_frame < frame_start:
                f.readline() # Simply skip and not parse the lines with frame < frame_start
                current_frame += stride
                continue
            # Reached the desired frame
            for i in range(frame_end - frame_start):
                line = f.readline()
                split_line = line.split(',')[:-1]
                agg_frame = list(map(int, split_line))
                cluster_agg.append(agg_frame)
                current_frame += stride
    return cluster_agg

# Generates aggregation histograms for a list of lists with agggregation numbers of different frames
def generate_agg_histograms(agg_data, max_val, bin_width=2):
    bin_edges = np.arange(0, max_val + bin_width, bin_width)
    bin_centers = bin_edges[:-1] + (bin_width / 2)
    hist_counts = []
    for line_data in agg_data:
        if len(line_data) == 0:
            counts = np.zeros_like(bin_centers)
        else:
            counts, _ = np.histogram(line_data, bins=bin_edges)
        hist_counts.append(counts)
    return bin_centers, hist_counts



# ----------------------------------------
# ------IMPORT SPECIFIC PARTICLES---------
# ----------------------------------------

def import_part_types(part_file):
    part_types = []
    with open(part_file) as f:
        for line in f:
            part_types.append(int(line))
    return part_types

def import_traj_part_type(target_part_type, part_file, traj_file, frame_start = 1, frame_end = 100):
    part_types = import_part_types(part_file)
    n_particles = len(part_types)
    selected_indices = []
    for p, p_type in enumerate(part_types):
        if p_type == target_part_type:
            selected_indices.append(p)
    traj_part_type = []
    
    with open(traj_file) as f:
        while True:
            header = f.readline()
            if not header:
                break # End of file!
            frame, total_frames = map(int, header.strip().split(','))
            if frame > frame_end:
                break # Stop reading after end of specified frames range
            if frame < frame_start:
                for _ in range(n_particles):
                    f.readline() # Simply skip and not parse the lines with frame < frame_start
                continue
            frame_coords = []
            for i in range(n_particles):
                line = f.readline()
                if i in selected_indices:
                    coords = tuple(map(float, line.split(',')))
                    frame_coords.append(coords)
            traj_part_type.append(frame_coords)
        traj_part_type = np.array(traj_part_type)
    return traj_part_type



# ----------------------------------------
# -----------STRUCTURE FACTOR-------------
# ----------------------------------------

def structure_factor(positions, box_length, qmax=None, nq=100):
    positions = np.asarray(positions)
    n, dim = positions.shape
    # Spacing in reciprocal space
    dq = 2 * np.pi / box_length
    if qmax is None:
        qmax = 10 * dq
    # Integer reciprocal lattice vectors
    nmax = int(np.ceil(qmax / dq))
    q_vectors = []
    q_magnitudes = []
    
    # Generate reciprocal vectors
    ranges = [range(-nmax, nmax + 1)] * dim
    for nvec in np.array(np.meshgrid(*ranges)).T.reshape(-1, dim):
        if np.all(nvec == 0):
            continue
        qvec = dq * nvec
        qmag = np.linalg.norm(qvec)
        if qmag <= qmax:
            q_vectors.append(qvec)
            q_magnitudes.append(qmag)

    q_vectors = np.array(q_vectors)
    q_magnitudes = np.array(q_magnitudes)
    # Density modes rho(q)
    rho_q = np.exp(-1j * positions @ q_vectors.T).sum(axis=0)
    # Structure factor
    s_q = (np.abs(rho_q) ** 2) / n
    # Binning by |q|
    bins = np.linspace(0, qmax, nq + 1)
    qvals = 0.5 * (bins[:-1] + bins[1:])
    sq = np.zeros(nq)
    counts = np.zeros(nq)

    inds = np.digitize(q_magnitudes, bins) - 1
    for i, s in zip(inds, s_q):
        if 0 <= i < nq:
            sq[i] += s
            counts[i] += 1
    mask = counts > 0
    sq[mask] /= counts[mask]
    return qvals[mask], sq[mask]

def structure_factor_frames(positions_frames, box_length, qmax=None, nq=100):
    sq_frames = []
    for positions in positions_frames:
        q, sq = structure_factor(positions, box_length, qmax, nq)
        sq_frames.append(sq)
    sq_frames = np.array(sq_frames)
    sq_avg = np.average(sq_frames, axis = 0)
    sq_std = np.std(sq_frames, axis = 0)
    return q, sq_avg, sq_std



# ----------------------------------------
# -------------TRANSITIONS----------------
# ----------------------------------------

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

# Calculates matrix of oligomeric transitions for selected frame range
def calculate_ass_diss_matrices(associations, dissociations, first_frame, last_frame):
    max_ass_component = 0
    for ass in associations:
        if first_frame <= ass[0] <= last_frame:
            candidate = np.max(ass[1])
            if candidate > max_ass_component:
                max_ass_component = candidate
    if max_ass_component == 0:
        associations_matrix = []
    max_diss_component = 0
    for diss in dissociations:
        if first_frame <= diss[0] <= last_frame:
            candidate = np.max(diss[1])
            if candidate > max_diss_component:
                max_diss_component = candidate
    if max_diss_component == 0:
        dissociations_matrix = []
    if max_ass_component > 0 or max_diss_component > 0:
        associations_matrix = np.zeros((max_ass_component, max_ass_component), dtype = int)
        dissociations_matrix = np.zeros((max_diss_component, max_diss_component), dtype = int)

        for ass in associations:
            row = ass[1][0]-1
            col = ass[1][1]-1
            associations_matrix[row, col] += 1
            associations_matrix[col, row] += 1
        for diss in dissociations:
            row = diss[2][0]-1
            col = diss[2][1]-1
            dissociations_matrix[row, col] += 1
            dissociations_matrix[col, row] += 1
        
    return associations_matrix, dissociations_matrix

# Sums association/dissociation matrices
def sum_ass_diss_matrices(matrices):
    largest_size = 0
    for matrix in matrices:
        size = np.shape(matrix)
        if size[0] > largest_size:
            largest_size = size[0]
    sum_matrix = np.zeros((largest_size, largest_size))
    for i in range(len(sum_matrix)):
        line = sum_matrix[i]
        for j in range(len(line)):
            for matrix in matrices:
                try:
                    sum_matrix[i,j] += matrix[i,j]
                except:
                    pass
    return sum_matrix



# ----------------------------------------
# ----------CONDENSATE BUILD--------------
# ----------------------------------------

def calculate_distance_sq(box_limits, vec1, vec2):
    dist_squared = 0.0
    for i in range(len(vec1)):
        diff1 = (vec2[i] - vec1[i])**2
        diff2 = (vec2[i] - (vec1[i] + 2*box_limits))**2
        diff3 = (vec2[i] - (vec1[i] - 2*box_limits))**2
        dist_squared += min([diff1, diff2, diff3])
    return dist_squared

def assemble_distance_matrix_sq(box_limits, coordinates):
    n = len(coordinates)
    dist_matrix_sq = np.full((n,n), np.nan)
    for i in range(len(coordinates)):
        for j in range(i, len(coordinates)):
            if i == j:
                continue
            else:
                dist_sq = calculate_distance_sq(box_limits, coordinates[i], coordinates[j])
                dist_matrix_sq[i,j] = dist_sq
    return dist_matrix_sq

def build_condensates(distances, threshold):
    n = distances.shape[0]
    visited = set()
    condensates = []
    close = distances < threshold
    for i in range(n):
        if i in visited:
            continue
        stack = [i]
        condensate = []
        while len(stack) > 0:
            p = stack.pop()
            if p in visited:
                continue
            visited.add(p)
            condensate.append(p)
            neighbors = np.where(close[p])[0]
            for n in neighbors:
                if n not in visited:
                    stack.append(n)
        if len(condensate) > 1:
            condensates.append(condensate)
    return condensates

def dot_product(vec1, vec2):
    dot = 0
    for i in range(len(vec1)):
        dot += vec1[i] * vec2[i]
    return dot

def cross_product(vec1, vec2):
    i = vec1[1]*vec2[2] - vec1[2]*vec2[1]
    j = vec1[2]*vec2[0] - vec1[0]*vec2[2]
    k = vec1[0]*vec2[1] - vec1[1]*vec2[0]
    cross = np.array([i, j, k])
    return cross

# Input: list of four points
def volume_tetrahedron(points):
    a = points[1] - points[0]
    b = points[2] - points[0]
    c = points[3] - points[0]
    cross = cross_product(a,b)
    dot = dot_product(cross, c)
    volume = 1/6 * np.abs(dot)
    return volume

# Volume of condensate with cutoff to avoid convex hulls
def volume_condensate(points, box_limit, cutoff = 1e8):
    cutoff_sq = cutoff**2
    delaunay = Delaunay(points)
    volume = 0
    for simplex in delaunay.simplices:
        skip_simplex = False
        vol_add = 0
        for p1 in range(len(simplex)):
            if skip_simplex == True:
                continue
            else:
                for p2 in range(1, len(simplex)):
                    coord1 = points[simplex[p1]]
                    coord2 = points[simplex[p2]]
                    l_sq = calculate_distance_sq(box_limit, coord1, coord2)
                    if l_sq > cutoff_sq:
                        vol_add = 0
                        skip_simplex = True
                        continue
                    else:
                        vol_add = volume_tetrahedron(points[simplex])
        volume += vol_add
    return volume