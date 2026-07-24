import numpy as np
from scipy.spatial import Delaunay

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