## RMSF-net

### System requirements

We recommened Linux system, with GPUs or CPUs support (This script is for CPU version, simply editing will be suitable for GPU support).

### Install

1. Downlaod and install UCSF chimera from  https://www.cgl.ucsf.edu/chimera/download.html, and set environment variable "CHIMERA_PATH" as the path to chimera exeutable file, e.g., "path_to_chimera/bin/chimera".

2. Install Anaconda or Miniconda.

3. Run ./install.sh, to create a python environment called rmsf-net. 



### Usage
For flexibility prediction, execute command:
```
conda activate rmsf-net
python predict.py -p pdb_file -e emd_file -o output_dir -c contour_level -m mode 

```

- pdb_file: The user-uploaded PDB file in .pdb format.
- emd_file: The user-uploaded cryo-EM map in .map or .mrc format.
- output_dir: The path for prediction output, can be manually specified.
- mode: The mode selected by the user, which can be 1, 2, or 3, corresponding to the three options "Only Cryo-EM","Only PDB model","Cryo-EM plus PDB model"

We recommened to use mode 3, which is the main method in our work.

Example:
```
python predict.py -p "data/6FBV.pdb" -e "data/emd_4230.map" -o "results" -m 3 
```

If you can not set chimera environment variable , you can also specify the chimera exeutable path at the command line, as following: 

```
python predict.py -p "data/6FBV.pdb" -e "data/emd_4230.map" -o "results" -m 3 -ch "path_to_chimera/bin/chimera"
```
 


#### Output
Run the command line above, the program will generate PDB simulated map (sim_map.mrc) and data file (data.pth) and save to "./results" directory. 

The predicted RMSF will be normalized and mapped onto the bfactor column of the original PDB file, saved to  "./results" directory according to the mode.  

You can visualize the RMSF output using the output pdb file at Chimera or Pymol.





