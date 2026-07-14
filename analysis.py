import numpy as np

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
        association_matrix = []
    max_diss_component = 0
    for diss in dissociations:
        if first_frame <= diss[0] <= last_frame:
            candidate = np.max(diss[1])
            if candidate > max_diss_component:
                max_diss_component = candidate
    if max_diss_component == 0:
        dissociation_matrix = []
    if max_ass_component > 0 or max_diss_component > 0:
        associations_matrix = np.zeros((max_ass_component, max_ass_component), dtype = int)
        dissociations_matrix = np.zeros((max_diss_component, max_diss_component), dtype = int)

        for ass in associations:
            row = ass[1][0]-1
            col = ass[1][1]-1
            associations_matrix[row, col] += 1
        for diss in dissociations:
            row = diss[2][0]-1
            col = diss[2][1]-1
            dissociations_matrix[row, col] += 1
        
    return associations_matrix, dissociations_matrix