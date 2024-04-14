from predict import *


def web_process(pdb_file, map_file, contour_level, mode, chimera_path=None ,data_file=None, output_dir=None):
    print(pdb_file, map_file, contour_level, mode, data_file, output_dir)

    # if output_dir is None:
    #     output_dir = ...
    mode=int(mode)
    pre = predict_map(pdb_file,
                      map_file,
                      output_dir, chimera_path=chimera_path, data_file=data_file, contour_level=contour_level, mode=mode)
    pre.predict()

    if mode == 1:
        out_file = f"{output_dir}/only_pdb/rmsf_nor.pdb"
    elif mode == 2:
        out_file = f"{output_dir}/only_cryoem/rmsf_nor.pdb"
    elif mode == 3:
        out_file = f"{output_dir}/pdb_plus_cryoem/rmsf_nor.pdb"

    return out_file

if __name__=="__main__":

    pdb_file="./data/7PQQ.pdb"
    map_file= "./data/emd_13596.map"
    contour_level=0.2
    mode=2
    output_dir="./data/predict"
    re=web_process(pdb_file, map_file, contour_level, mode, data_file=None, output_dir=output_dir)