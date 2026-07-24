# kinetic_patchy_particles (cool name coming soon - I hope!)
# About #
This program performs Monte Carlo simulations for patchy particle systems using Brownian dynamics. Patchy particles are colloidal models commonly employed on modelling self-assembly processes, such as protein aggregation and phase separation. This program is a project in progress for studying increasingly complex systems, specifying different interaction affinities between patches (interaction sites), having different particle types in a simulation, such as systems of crowders, ions and proteins.
In the current implementation, patchy particles are hard spheres which interact through its patches by a square well potential.
This was written with as little dependencies as possible, using only very standard libraries such as Numpy, Scipy and Matplotlib. All other imports are from different moduli contained in this repository.

# Usage #
This repository contains some function libraries and a script (run.py) which runs the actual simulation. For that, you simply need to run the command on the directory with all the scripts:
`python run.py simulation_params.par particle_params.par interaction_params.par`
And that's it! Then you just have to wait until it's complete. You can insert different flags, such as specifying prefixes and folders for the output files. These pieces of information are displayed when typing `python run.py -h`

# Getting involved #
If you would like to contribute to this project, are having trouble running the scripts, want to know more about the code or simply have some ideas on future implementations, don't hesitate to send a message! In the future there will be more in-depth tutorials.

# Disclaimer #
The project was written with very little AI use, specifically for a few bugs encountered during testing, and by someone who is not a professional programmer, but a biophysics researcher. Therefore, the code is not as optimized or elegantly coded as well-established pieces of software. Still, I hope this helps whoever has the need to understand processes of self-assembly!
