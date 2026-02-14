import numpy as np
import torch
import mrcfile
from scipy import ndimage
import subprocess
import sys
from moleculekit.molecule import Molecule
import os


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


def parse_map2(map):

    mrc = mrcfile.open(map, 'r')

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
    map2 = np.asfarray(mrc.data.copy(), dtype=np.float32)

    assert (angle[0] == angle[1] == angle[2] == 90.0)
    mapcrs = np.subtract(
        [mrc.header.mapc, mrc.header.mapr, mrc.header.maps], 1)
    sort = np.asarray([0, 1, 2], dtype=np.int32)
    if (mapcrs == sort).all():
        changed = False
        xyz_start = np.asarray([start_xyz[i] for i in sort])
        nxyz = np.asarray([ncrs[i] for i in sort])
        mrc.close()
    else:
        changed = True
        for i in range(3):
            sort[mapcrs[i]] = i
        xyz_start = np.asarray([start_xyz[i] for i in sort])
        nxyz = np.asarray([ncrs[i] for i in sort])
        map2 = np.transpose(map2, axes=2-sort[::-1])
        mrc.close()

    info = dict()
    info['cella'] = cella
    info['xyz_start'] = xyz_start
    info['voxel_size'] = voxel_size
    info['nxyz'] = nxyz
    info['origin'] = origin
    info['changed'] = changed

    return map2, info


def get_atom_map(pdb_file, shape, map_info, r=1.5):

    atom_map = np.full((shape[0], shape[1], shape[2]), 0, dtype=np.int8)
    pdb = Molecule(pdb_file)
    pdb.filter('protein')
    xyz = pdb.get('coords')-map_info['origin']
    xyz_norm = ((xyz-map_info['voxel_size']*map_info['xyz_start'])/r)
    for coord in xyz_norm:
        atom_map[int(coord[2]), int(coord[1]), int(coord[0])] = 1
    return atom_map


def split_map_and_select_item(map, atom_map, contour_level, box_size=40, core_size=10):

    map_size = np.shape(map)
    pad_map = np.full((map_size[0]+2*box_size, map_size[1] +
                      2*box_size, map_size[2]+2*box_size), 0, dtype=np.float32)
    pad_map[box_size:-box_size, box_size:-box_size, box_size:-box_size] = map

    pad_atom_map = np.full(
        (map_size[0]+2*box_size, map_size[1]+2*box_size, map_size[2]+2*box_size), 0, dtype=np.int8)
    pad_atom_map[box_size:-box_size, box_size:-
                 box_size, box_size:-box_size] = atom_map

    start_point = box_size - int((box_size - core_size) / 2)

    cur_x, cur_y, cur_z = start_point, start_point, start_point

    box_list = list()

    length = [int(np.ceil(map_size[0]/core_size)),
              int(np.ceil(map_size[1]/core_size)), int(np.ceil(map_size[2]/core_size))]
    print(
        f"the total box of this map is {length[0]}*{length[1]}*{length[2]}={length[0]*length[1]*length[2]}")
    keep_list = []
    total_list = []
    i = 0
    while (cur_z + (box_size - core_size) / 2 < map_size[2] + box_size):

        next_box = pad_map[cur_x:cur_x + box_size,
                           cur_y:cur_y + box_size, cur_z:cur_z + box_size]
        next_atom_box_center = pad_atom_map[cur_x+15:cur_x + box_size -
                                            15, cur_y+15:cur_y + box_size-15, cur_z+15:cur_z + box_size-15]

        cur_x += core_size
        if (cur_x + (box_size - core_size) / 2 >= map_size[0] + box_size):
            cur_y += core_size
            cur_x = start_point
            if (cur_y + (box_size - core_size) / 2 >= map_size[1] + box_size):
                cur_z += core_size
                cur_y = start_point
                cur_x = start_point

        if (np.sum(next_atom_box_center) > 0):

            box_list.append(next_box)

            keep_list.append(i)
        total_list.append(i)
        i = i+1
    print(f"the selected maps: {len(keep_list)}")
    print(f"the total maps: {len(total_list)}")
        
    # Limit to first 70 boxes for testing
    box_list = box_list[:70]
    keep_list = keep_list[:70]
    total_list = total_list[:70]

    return np.asarray(box_list), np.asarray(keep_list), np.asarray(total_list)

#import subprocess

def get_smi_map(pdb_file, res, out_file, chimera_path=None, number=0.1, r=1.5):
    # Create ChimeraX command file
    with open('./chimera_exe.cxc', 'w') as chimera_script:
        chimera_script.write(
            f'open {pdb_file}\n'
            f'molmap #1 {res} gridSpacing {r}\n'
            f'volume #2\n'
            f'save {out_file}\n'
            'close all\n'
        'exit\n'
        )

    print(f'chimera_path: {chimera_path}')

    try:
        output = subprocess.check_output(
            [chimera_path, '--nogui', './chimera_exe.cxc'],
            stderr=subprocess.STDOUT
        )
        decoded_output = output.decode() if output else ""
        print("ChimeraX output:")
        print(decoded_output)
        return decoded_output
    except subprocess.CalledProcessError as e:
        print("ChimeraX failed with error:")
        error_output = e.output.decode() if e.output else "No output captured."
        print(error_output)
        return None


# def sim_map_ot(pdb_file, res, out_file, number=0.1, r=1.5, chimera_path=None):
def sim_map_ot(pdb_file, res, out_file, number=2, r=1.5, chimera_path=None):
    output = get_smi_map(pdb_file, res, out_file, chimera_path=chimera_path, r=r)

    if output is None:
        return False

    s = output.splitlines()
    if not s or "Wrote file" not in s[-1]:
        output = get_smi_map(pdb_file, res, out_file, r=r, number=1, chimera_path=chimera_path)
        if output is None:
            return False
        s = output.splitlines()
#        if not s or "Wrote file" not in s[-1]:
#            return False
        if os.path.exists(out_file):
                return True
        else:
                print(f"Output file {out_file} not found.")

        return False

    return True


class exp_2_sim:
    def __init__(self, exp_map_file, sim_map_file, keep_list, total_list):
        self.sim_map, self.sim_info = parse_map2(sim_map_file)
        self.exp_map, self.exp_info = parse_map(exp_map_file)
        self.keep_list = keep_list
        self.total_list = total_list

    @staticmethod
    def convert_to_n_base(number, n_3):
        result = []
        i = 0
        while number > 0:
            remainder = number % n_3[i]
            result.append(remainder)
            number = number // n_3[i]
            i = i+1
        if len(result) != 3:
            if len(result) == 2:
                result.append(0)
            elif len(result) == 1:
                result.extend([0, 0])
            elif len(result) == 0:
                result.extend([0, 0, 0])
            else:
                print(f"too big,bigger than 3 digits")
        assert (len(result) == 3)
        result.reverse()

        return result

    def index_start(self, index, box_size=40, core_size=10, step_size=40):

        start_point = box_size - int((box_size - core_size) / 2)
        cur_box_num = self.keep_list[index]

        exp_shape = self.exp_map.shape
        num_per_dim = [(exp_shape[0]+9)//10, (exp_shape[1]+9) //
                       10, (exp_shape[2]+9)//10]

        box_num_zyx = self.convert_to_n_base(int(cur_box_num), num_per_dim)

        x11, y11, z11 = box_num_zyx[2]*core_size+start_point-box_size, box_num_zyx[1]*core_size+start_point-box_size,\
            box_num_zyx[0]*core_size+start_point-box_size

        add_center = int((box_size-step_size)/2)

        x11, y11, z11 = x11+add_center, y11+add_center, z11+add_center
        exp_index = [x11, y11, z11]

        return exp_index

    def trans_index_exp2sim(self, exp_index):
        ''' convert indices on the experimental map to indices on the simulated map.'''
        x, y, z = exp_index[0], exp_index[1], exp_index[2]

        x_coord, y_coord, z_coord = self.exp_info['origin'] + \
            self.exp_info['xyz_start'] * \
            self.exp_info['voxel_size']+np.array([z, y, x])*1.5

        xyz_sim_index = [round(i) for i in reversed((np.array([x_coord, y_coord, z_coord]) -
                                                     self.sim_info['origin']-self.sim_info['voxel_size']*self.sim_info['xyz_start'])/1.5)]

        return xyz_sim_index


    def range2range_axis(self, x_left, x_right, i=0):

        if x_left < 0 and x_right < 0:
            print("out of sim_map,no overlap")
            return 0, 0, 0, 0
        if x_left > self.sim_info['nxyz'][i] and x_right > self.sim_info['nxyz'][i]:
            print("out of sim_map,no overlap")
            return 0, 0, 0, 0
        if x_left >= 0 and x_right < self.sim_info['nxyz'][i]:
            x_to_left, x_to_right = 0, x_right-x_left
        elif x_left < 0 and x_right < self.sim_info['nxyz'][i]:
            x_to_left, x_to_right = -x_left, x_right-x_left
            x_left = 0
        elif x_left >= 0 and x_right >= self.sim_info['nxyz'][i]:
            x_to_left, x_to_right = 0, self.sim_info['nxyz'][i]-x_left
            x_right = self.sim_info['nxyz'][i]
        elif x_left < 0 and x_right >= self.sim_info['nxyz'][i]:
            x_to_left, x_to_right = -x_left, self.sim_info['nxyz'][i]-x_left
            x_left = 0
            x_right = self.sim_info['nxyz'][i]
        return x_left, x_right, x_to_left, x_to_right
    

    def trans_range2range(self, xyz_sim_index, box_size=40, pad=0.):

        x_left, x_right, x_to_left, x_to_right = self.range2range_axis(
            xyz_sim_index[0], xyz_sim_index[0]+box_size, i=2)
        y_left, y_right, y_to_left, y_to_right = self.range2range_axis(
            xyz_sim_index[1], xyz_sim_index[1]+box_size, i=1)
        z_left, z_right, z_to_left, z_to_right = self.range2range_axis(
            xyz_sim_index[2], xyz_sim_index[2]+box_size, i=0)

        return [x_left, x_right], [y_left, y_right], [z_left, z_right], [x_to_left, x_to_right], [y_to_left, y_to_right], [z_to_left, z_to_right]


    def gene_boxs(self, box_size=40, pad=0.):

        sim_box_list = []
        for index in range(len(self.keep_list)):

            exp_index = self.index_start(index)
            xyz_sim_index = self.trans_index_exp2sim(exp_index)
            sim_x_lr, sim_y_lr, sim_z_lr, box_x_lr, box_y_lr, box_z_lr = self.trans_range2range(
                xyz_sim_index)

            sim_box = np.full([box_size, box_size, box_size],
                              pad, dtype=np.float32)

            sim_box[box_x_lr[0]:box_x_lr[1], box_y_lr[0]:box_y_lr[1], box_z_lr[0]:box_z_lr[1]
                    ] = self.sim_map[sim_x_lr[0]:sim_x_lr[1], sim_y_lr[0]:sim_y_lr[1], sim_z_lr[0]:sim_z_lr[1]]

            sim_box_list.append(sim_box)

        sim_box_list = np.array(sim_box_list)

        return sim_box_list


class gen_data:
    def __init__(self, exp_map_file, pdb_file, output_dir, contour_level, chimera_path=None) -> None:
        self.exp_map_file = exp_map_file
        self.pdb_file = pdb_file
        self.output_dir = output_dir
        self.contour_level = contour_level
        self.chimera_path = chimera_path

    def get_data(self, r=1.5):
        print("Starting data generation in gen_data.get_data()...")

        sim_map_file = f"{self.output_dir}/sim_map.mrc"

        map, info = parse_map(self.exp_map_file, r=r)
        print("Parsed experimental map and generated atom map.")
        atom_map = get_atom_map(self.pdb_file, map.shape, info)

        print("Split map and selected items.")
        intensity_list, keep_list, total_list = split_map_and_select_item(
            map, atom_map, self.contour_level, box_size=40, core_size=10)
        print("Calling ChimeraX to generate simulated map...")

        sim_yes = sim_map_ot(self.pdb_file, 4, sim_map_file,
                             r=r, chimera_path=self.chimera_path)
        print("Simulated map generated successfully.")
        if not sim_yes:
            print(" sim map not succeed")
            sys.exit()

        expsim = exp_2_sim(self.exp_map_file, sim_map_file,
                           keep_list, total_list)
        sim_box_list = expsim.gene_boxs()

        data_file = f"{self.output_dir}/data.pth"
        print("Saving tensor data to .pth file...")

        torch.save({'intensity': torch.from_numpy(intensity_list).unsqueeze_(1), 'sim_intensity': torch.from_numpy(sim_box_list).unsqueeze_(1),
                    'keep_list': torch.from_numpy(keep_list), 'total_list': torch.from_numpy(total_list)}, data_file)
    
        print(f"tensor datafile saved at {data_file}")
        return data_file
