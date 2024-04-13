### System requirements

We recommened Linux system, with GPUs or CPUs support (This script is for CPU version, simply editing will be suitable for GPU support).

### Install

1. Downlaod and install UCSF chimera from  https://www.cgl.ucsf.edu/chimera/download.html, and set environment variable "CHIMERA_PATH" to the path to chimera exeutable file, e.g., "path_to_chimera/bin/chimera".

2. install Anaconda or Miniconda

3. simple run ./install.sh


### Usage
python predict.py -p pdb_file -e emd_file -o output_dir -c contour_level -m mode

- pdb_file: The user-uploaded PDB file in .pdb format.
- emd_file: The user-uploaded cryo-EM map in .map or .mrc format.
- output_dir: The path for prediction output, can be manually specified, default is ./data/predict.
- contour_level: The user-defined contour level.
- mode: The mode selected by the user, which can be 1, 2, or 3, corresponding to the three options "Only Cryo-EM","Only PDB model","Cryo-EM plus PDB model"

We recommened to use mode 3, which is the main method in our work.
Example: python predict.py -p "./data/7PQQ.pdb" -e "./data/emd_13596.map" -m 3 -c 0.2

#### Output
Run the command line above, the program will generate PDB simulated map (sim_map.mrc) and data file (data.pth) and save to "./results" directory. 

The final predicted RMSF will be mapped onto the bfactor column of the original PDB file, saved to  "./results" directory according to the mode.  

You can visualize the RMSF output using the output pdb file at Chimera or Pymol.





