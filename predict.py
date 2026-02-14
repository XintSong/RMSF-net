mport torch
import torch.nn as nn
import torch.functional as F
# import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
import argparse
from model import RMSF_net_model
from moleculekit.molecule import Molecule
import mrcfile
from gen_data import *

import os



def get_parser():
    parser = argparse.ArgumentParser(
        description='Prediction for RMSF',
        usage=f"python predict.py -p pdbfile -e emdfile -d datafile -o outputdir -c contour_level"
    )

    parser.add_argument(
        '-p', '--pdb', type=str,
        help="path to pdbfile"
    )
    parser.add_argument(
        "-e", "--emd", type=str,
        help="path to cryo_em map"
    )
    parser.add_argument(
        "-d", "--data", type=str, default=None,
        help="data file ,default mean generate from scratch "
    )
    parser.add_argument(
        "-o", "--out_dir", type=str, default="./data/predict",
        help="path to output dir"
    )

    parser.add_argument(
        "-ch", "--chimera_path", type=str,
        help="chimera_path"
    )

    parser.add_argument(
        "-c", "--contour", type=str,
        help="contour level"
    )

    parser.add_argument(
        "-m", "--mode", type=str, default=None,
        help="mode choice"
    )

    return parser.parse_args()


def normalize_and_avg_rmsf(pdb, target_pdb):
    #print(pdb, target_pdb)
    mol = Molecule(pdb)
    mol.filter('protein')
    bf = mol.beta
    non_zero_bf = np.array(list(filter(lambda x: x != 0, bf)))

    mean = np.mean(non_zero_bf)
    std = np.std(non_zero_bf)

    normalized_bf = (non_zero_bf - mean) / std

    non_zero_indices = np.nonzero(bf)

    bf[non_zero_indices] = normalized_bf
    rid_chid = [f'{r}{c}' for r, c in zip(
        mol.get('resid', sel='name CA'), mol.get('chain', sel='name CA'))]
    rid_chids = [f'{rs}{cs}' for rs, cs in zip(
        mol.get('resid'), mol.get('chain'))]
    for i in rid_chid:
        identical_idx = [n for n, x in enumerate(rid_chids) if x == i]
        beta_res = bf[identical_idx]
        beta_res_nonzero = beta_res[beta_res != 0]
        if len(beta_res_nonzero) != 0:
            avr_res_bf = np.mean(beta_res_nonzero)
        else:
            avr_res_bf = 0
        bf[identical_idx] = avr_res_bf
    mol.beta = bf
    mol.write(target_pdb)
    print(f'RMSF Averaged and Normalized on {target_pdb}')


def box2map(ana_pre, keep_list, info, box_size=40, core_size=10, pad=None):

    # print(ana_pre.shape)
    csize = (ana_pre.shape[-1]-core_size)//2

    if csize == 0:
        ana_box_list = ana_pre
    else:
        ana_box_list = ana_pre[:, csize:-csize, csize:-csize, csize:-csize]
    map_size = info['nxyz']
    edge_size = (box_size-core_size)//2

    if not pad:
        pad = 0
    pad_map = np.full((map_size[0]+2*box_size, map_size[1]+2 *
                      box_size, map_size[2]+2*box_size), pad, dtype=np.float32)

    start_point = box_size - int((box_size - core_size) / 2)
    cur_x, cur_y, cur_z = start_point, start_point, start_point

    i = 0
    cur_pos = 0
    if True:
        while (cur_z + (box_size - core_size) / 2 < map_size[2] + box_size):

            cur = keep_list[cur_pos]
            if i == cur:

                next_ana_box = ana_box_list[cur_pos]

                pad_map[cur_x+edge_size:cur_x + box_size-edge_size, cur_y+edge_size:cur_y + box_size -
                        edge_size, cur_z+edge_size:cur_z + box_size-edge_size] = next_ana_box  # 这里要取中间10的方块
                cur_pos = cur_pos+1
                if cur_pos >= len(keep_list):
                    break
            i = i+1

            cur_x += core_size
            if (cur_x + (box_size - core_size) / 2 >= map_size[0] + box_size):
                cur_y += core_size
                cur_x = start_point
                if (cur_y + (box_size - core_size) / 2 >= map_size[1] + box_size):
                    cur_z += core_size
                    cur_y = start_point
                    cur_x = start_point

    ana_map = pad_map[box_size:-box_size,
                      box_size:-box_size, box_size:-box_size]

    return ana_map


def write_rmsf_pdb(pdb_file, ana_map, info, save_dir, r=1.5, protein=True):

    mol = Molecule(pdb_file)
    if protein:
        mol.filter('protein')
    xyz = mol.get('coords')-info['origin']

    xyz_norm = ((xyz-info['voxel_size']*info['xyz_start'])/r)

    ana_list = []
    for cor in xyz_norm:

        try:
            x, y, z = int(cor[2]), int(cor[1]), int(cor[0])

            ana_list.append(ana_map[x, y, z])
        except:

            ana_list.append(0.)
            continue
    ana_list = np.array(ana_list)

    # ana_list=(ana_list-np.min(ana_list))/(np.max(ana_list)-np.min(ana_list))
    mol.set('beta', np.array(ana_list))
    save_file = f"{save_dir}/rmsf.pdb"
    mol.write(save_file)
    # print(f"the pdb visualization file saved at {save_file}")
    return save_file


def onehot(sse):
    sse_ = sse
    sse_ = F.one_hot(sse_).permute(0, 4, 1, 2, 3)
    return sse_


def generate_mask(ana_box_list, mask=0, rmsf=True, tensor=True):

    if tensor:
        if rmsf:
            mask_list = torch.where(ana_box_list != mask, 1, 0)
        else:
            mask_list = torch.where(ana_box_list == mask, 0, 1)
    else:
        if rmsf:
            mask_list = np.where(ana_box_list != mask, 1, 0)
        else:
            mask_list = np.where(ana_box_list == mask, 0, 1)
    return mask_list





def parse_map(map_file, r=1.5):
    mrc = mrcfile.open(map_file, 'r')

    voxel_size = np.asarray(
        [mrc.voxel_size.x, mrc.voxel_size.y, mrc.voxel_size.z], dtype=np.float32)
    cella = (mrc.header.cella.x, mrc.header.cella.y, mrc.header.cella.z)
    origin = np.asarray([mrc.header.origin.x, mrc.header.origin.y,
                        mrc.header.origin.z], dtype=np.float32)

    start_xyz = np.asarray(
        [mrc.header.nxstart, mrc.header.nystart, mrc.header.nzstart], dtype=np.float32)

    ncrs = (mrc.header.nx, mrc.header.ny, mrc.header.nz)
    angle = np.asarray([mrc.header.cellb.alpha, mrc.header.cellb.beta,
                       mrc.header.cellb.gamma], dtype=np.float32)
    map = np.asfarray(mrc.data.copy(), dtype=np.float32)

    assert (angle[0] == angle[1] == angle[2] == 90.0)
    mapcrs = np.subtract(
        [mrc.header.mapc, mrc.header.mapr, mrc.header.maps], 1)
    sort = np.asarray([0, 1, 2], dtype=np.int32)
    for i in range(3):
        sort[mapcrs[i]] = i
    xyz_start = np.asarray([start_xyz[i] for i in sort])
    nxyz = np.asarray([ncrs[i] for i in sort])
    map = np.transpose(map, axes=2-sort[::-1])
    mrc.close()

    zoomFactor = np.divide(voxel_size, np.asarray([r, r, r], dtype=np.float32))
    map2 = ndimage.zoom(map, zoom=zoomFactor)
    nxyz = np.asarray([map2.shape[0], map2.shape[1],
                      map2.shape[2]], dtype=np.int32)

    info = dict()
    info['cella'] = cella
    info['xyz_start'] = xyz_start
    info['voxel_size'] = voxel_size
    info['nxyz'] = nxyz
    info['origin'] = origin

    return map2, info


class predict_map:

    def __init__(self, pdb_file, map_file, output_dir, chimera_path=None, data_file=None, contour_level=None, mode=None) -> None:

        self.pdb_file = pdb_file
        self.map_file = map_file
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        if not data_file:
            if chimera_path is None:
                chimera_path = os.environ.get('CHIMERA_PATH')
            new_data = gen_data(map_file, pdb_file, output_dir,
                                contour_level, chimera_path=chimera_path)
            self.data_file = new_data.get_data()
        else:
            self.data_file = data_file

        self.output_dir = output_dir
        self.mode = mode

        # i=2
        self.model_file_sim = f'./model_cpu/sim_model.pth'
        self.model_file_exp_and_sim = f'./model_cpu/exp_sim_model.pth'
        self.model_file_exp = f'./model_cpu/exp_model.pth'

    # model,i,log_dir,data_dir,f

    def test_model_for_rmsf(self, data, model, model_file, img_dir=None, dsize=(40, 40, 10), flag=None, gt=False):
        device = torch.device('cpu')

        model.load_state_dict(torch.load(model_file))
        model.eval()

        if flag == "sim":
            test_data = data["sim"]
        elif flag == "exp":
            test_data = data["exp"]
        elif flag == "sim01":
            test_data = onehot(data["sim01"].long()).float()
        elif flag == "exp_and_sim":
            test_data = data["exp"]
            test_sim = data["sim"]

        if gt:
            cor = 0
            mean_loss = 0

            loss_fn1 = nn.MSELoss(reduction='sum')
            with torch.no_grad():

                print(f"* test rmsf *  \n \n")
                test_data, test_rmsf = test_data.to(
                    device), data["rmsf"].to(device)

                keep_list = data['keep_list']

                test_rmsf_mask = generate_mask(
                    test_rmsf, mask=0, rmsf=True, tensor=True)
                if flag == "exp_and_sim":
                    test_sim = test_sim.to(device)
                    model_rmsf = model(test_data, sse=test_sim)
                else:
                    model_rmsf = model(test_data)

                rmsf_pre = model_rmsf*(test_rmsf_mask)
                loss1 = loss_fn1(rmsf_pre, test_rmsf)
                # length_rmsf = torch.sum(test_rmsf_mask)
                # loss = loss1

                rmsf_pree = torch.take(
                    rmsf_pre.flatten(), torch.where(test_rmsf.flatten() != 0)[0])
                test_rmsff = torch.take(
                    test_rmsf.flatten(), torch.where(test_rmsf.flatten() != 0)[0])
                rmsf_preee = rmsf_pree.detach().numpy()
                test_rmsfff = test_rmsff.detach().numpy()
                exp_1, exp_2 = np.power(10, rmsf_preee) - \
                    1, np.power(10, test_rmsfff)-1

                cor = round(np.corrcoef(exp_1, exp_2)[0, 1], 3)

                print(f"cor is:{cor} \n \n")

                ana_list = rmsf_pre.squeeze(1).detach().numpy()
                gt_list = test_rmsf.squeeze(1).numpy()
                if flag == "exp_and_sim":
                    del test_sim
                del test_data, test_rmsf, test_rmsf_mask, rmsf_pre

                mean_loss = round(np.mean(np.abs(exp_1-exp_2)), 3)
                print(f" loss for rmsf is {mean_loss} \n ")

            return ana_list, gt_list, cor, keep_list, mean_loss, test_rmsfff, rmsf_preee

        else:

            with torch.no_grad():

                print(f"* predict rmsf *  \n \n")
                test_data = test_data.to(device)
                keep_list = data['keep_list']
                if flag == "exp_and_sim":
                    test_sim = test_sim.to(device)
                    model_rmsf = model(test_data, sse=test_sim)
                else:
                    print(test_data.shape)
                    model_rmsf = model(test_data)

                ana_list = model_rmsf.squeeze(1).detach().numpy()
                if flag == "exp_and_sim":
                    del test_sim
                del test_data

            return ana_list, keep_list

    def test_item(self, data, model_file, img_dir=None, dsize=(40, 40, 10), flag=None, gt=False):

        if flag == "sim" or flag == "exp":
            input_chan = 1
        elif flag == "sim01" or flag == "exp_and_sim":
            input_chan = 2

        model = RMSF_net_model(in_channels=input_chan)
        if gt:
            ana_list, gt_list, cor, keep_list, mean_loss, test_rmsfff, rmsf_preee = self.test_model_for_rmsf(
                data, model, model_file, img_dir=img_dir, dsize=dsize, flag=flag, gt=gt)

            print(
                f"**************cor for this protein for {flag} is {cor}*****************")
            return ana_list, gt_list, cor, keep_list, mean_loss, test_rmsfff, rmsf_preee
        else:
            ana_list, keep_list = self.test_model_for_rmsf(
                data, model, model_file, img_dir=img_dir, dsize=dsize, flag=flag, gt=gt)
            return ana_list, keep_list

    
    def predict(self, r=1.5, dsize=(40, 40, 10)):

        if self.mode is None:
            model_files = [self.model_file_sim,
                           self.model_file_exp, self.model_file_exp_and_sim]
            flags = ['sim', 'exp', 'exp_and_sim']
        else:
            if int(self.mode) == 1:
                model_files = [self.model_file_sim]
                flags = ['sim']
            elif int(self.mode) == 2:
                model_files = [self.model_file_exp]
                flags = ['exp']
            elif int(self.mode) == 3:
                model_files = [self.model_file_exp_and_sim]
                flags = ['exp_and_sim']
        flag_map = {'sim': 'only_pdb', 'exp': 'only_cryoem',
                    'exp_and_sim': 'pdb_plus_cryoem'}

        _, info = parse_map(self.map_file, r=r)

        data = {}
        datapth = torch.load(self.data_file)

        data["exp"] = datapth["intensity"]
        data["sim"] = datapth["sim_intensity"]

        data["keep_list"] = datapth["keep_list"]
        data["total_list"] = datapth["total_list"]

        if 'rmsf' in datapth.keys():
            data['rmsf'] = datapth['rmsf']
            gt = True
        else:
            gt = False

        if gt:

            gt_dir = f"{self.output_dir}/gt"
            if not os.path.exists(gt_dir):
                os.mkdir(gt_dir)
            for model_file, flag in zip(model_files, flags):
                sub_dir = f"{self.output_dir}/{flag}"
                if not os.path.exists(sub_dir):
                    os.mkdir(sub_dir)
                ana_list, gt_list, cor, keep_list, mean_loss, gt_rmsf, pre_rmsf = self.test_item(
                    data, model_file, img_dir=sub_dir, dsize=dsize, flag=flag, gt=gt)
                ana_map = box2map(ana_list, keep_list, info,
                                  box_size=40, core_size=10)
                ana_map = np.power(10, ana_map)-1
                write_rmsf_pdb(self.pdb_file, ana_map, info, sub_dir, r=r)
                # write_rmsf_map(map,ana_map,info,pre_dir,r=r)
                rmsf_list = np.array([gt_rmsf, pre_rmsf]).reshape(-1, 2)
                rmsf_list = np.power(10, rmsf_list)-1
                np.savetxt(f'{sub_dir}/rmsf_list.txt',
                           rmsf_list, fmt='%.3f', delimiter='\t')
                print(f'correlation coff is {cor}')
                print(f'mean loss is {mean_loss}')
            gt_map = box2map(gt_list, keep_list, info,
                             box_size=40, core_size=10)
            gt_map = np.power(10, gt_map)-1
            save_file = write_rmsf_pdb(
                self.pdb_file, gt_map, info, gt_dir, r=r)
            normalize_and_avg_rmsf(
                save_file, f'{os.path.dirname(save_file)}/rmsf_nor.pdb')
            os.remove(save_file)

        else:
            for model_file, flag in zip(model_files, flags):
                sub_dir = f"{self.output_dir}/{flag_map[flag]}"
                if not os.path.exists(sub_dir):
                    os.mkdir(sub_dir)
                ana_list, keep_list = self.test_item(
                    data, model_file, img_dir=sub_dir, dsize=dsize, flag=flag, gt=gt)
                # print(type(ana_list),type(keep_list))
                ana_map = box2map(ana_list, keep_list, info,
                                  box_size=40, core_size=10)
                ana_map = np.power(10, ana_map)-1
                save_file = write_rmsf_pdb(
                    self.pdb_file, ana_map, info, sub_dir, r=r)
                normalize_and_avg_rmsf(
                    save_file, f'{os.path.dirname(save_file)}/rmsf_nor.pdb')
                os.remove(save_file)


def main():
    print("Starting RMSF-net prediction...")

    args = get_parser()
    pdb_file = args.pdb
    map_file = args.emd
    data_file = args.data
    output_dir = args.out_dir
    if args.contour is not None:
        contour_level = float(args.contour)
    else:
        contour_level=None
    chimera_path=args.chimera_path
    mode = args.mode
    print(f"Parsed arguments: PDB={pdb_file}, MAP={map_file}, DATA={data_file}, OUT={output_dir}, CONTOUR={contour_level}, CHIMERA={chimera_path}, MODE={mode}")

    pre = predict_map(pdb_file,
                      map_file,
                      output_dir, data_file=data_file, chimera_path=chimera_path, contour_level=contour_level, mode=mode)

    # python predict.py -p "data/6FBV.pdb" -e "data/emd_4230.map" -o "results"  -c 0 -m 3
    print("Calling predict() method...")

    pre.predict()


if __name__ == "__main__":
    main()
