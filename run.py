import os
import shutil
import argparse
import utils
import simulation as sim
import time

start_time = time.perf_counter()

parser = argparse.ArgumentParser(prog='Monte Carlo Simulation',
                    description='MC simulation of patchy particles',
                    epilog='Hope this helps!')
parser.add_argument('params', help = 'Simulation parameters (.par) file') # positional argument
parser.add_argument('parts', help = 'Particle parameters (.par) file') # positional argument
parser.add_argument('inters', help = 'Interaction parameters (.par) file') # positional argument
parser.add_argument('-f', '--folder', default = 'outputs', help = 'Output folder (Default: outputs)')
parser.add_argument('-p', '--prefix', default = '', help = 'Prefix for output files (Default: None)')
parser.add_argument('-nt', '--no-transitions', help = 'Flag for disabling transitions calculations', action='store_false')
#parser.add_argument('-v', '--verbose', help = 'Verbose flag for less text', action='store_true')
args = parser.parse_args()
prefix_out = args.prefix
folder_out = args.folder
trans_out = args.no_transitions

# Parameter files from argparse
file_sim = args.params
file_parts = args.parts
file_inters = args.inters
files = [file_sim, file_parts, file_inters]
os.makedirs(folder_out, exist_ok=True)
for file in files:
    shutil.copy(file, folder_out)

# Convert parameter files into dictionaries
sim_params = utils.import_simulation_params(file_sim)
part_params = utils.import_particle_params(file_parts)
inter_params = utils.import_interaction_params(file_inters)

#paths = utils.create_files(prefix = prefix_out, folder = folder_out)
sim.run_simulation(sim_params, part_params, inter_params, prefix = prefix_out, folder = folder_out, transitions = trans_out)
end_time = time.perf_counter()
print(f"Finished in {(end_time - start_time)/60} minutes")

