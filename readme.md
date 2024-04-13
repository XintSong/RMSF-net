Usage: python predict.py -p pdb_file -e emd_file -o output_dir -c contour_level -m mode

- pdb_file: The user-uploaded PDB file in .pdb format.
- emd_file: The user-uploaded cryo-EM map in .map or .mrc format.
- output_dir: The path for prediction output, can be manually specified, default is ./data/predict.
- contour_level: The user-defined contour level.
- mode: The mode selected by the user, which can be 1, 2, or 3, corresponding to the three options from left to right.

Example: python predict.py -p "./data/7PQQ.pdb" -e "./data/emd_13596.map" -m 1 -c 0.2




