# Changed made

Original repository: https://github.com/XintSong/RMSF-net/

In my hand I could not run this software and documented my efforts in the [issues](https://github.com/XintSong/RMSF-net/issues) on the original repository.

ChimeraX versions tested worked from version `1.1.1` to version `1.10.1`. (Version `1.0` did not launch and could not be tested.)

Briefly the changes made to `predict.py` and `gen_data.py` are listed below. Those changes were implemented with the help of Microsoft Copilot [copilot.microsoft.com](copilot.microsoft.com)

## Changes in `predict.py`
A few debugging `print()` statements were added to help find issues.

## Changes in `gen_data.py`

- changed file name `chimera_exe.cmd` to `chimera_exe.cxc` file. This is necessary for running on macOS as the ChimeraX commands files should end in `.cxc`. The `.cmd` may possibly work on other systems.
- added `exit` final command to `chimera_exe.cxc`. Without this final command the system simply "hangs" forever.  
- changed the parameters of definition of function `sim_map_ot`: changed `number=0.1` to `number=2` which is the correct model number within ChimeraX for the `model #` in the command script.  
- Another change was with lines `if not s or "Wrote file" not in s[-1]:` which test if ChimeraX sends a "Write Out" information which failed. Copilot suggested to test the presence of the written file instead, as implemented here.
- In definition `def get_data(self, r=1.5):` multiple `print()` statements were added for debugging purpose and checkpoint tests.

The definition of function `split_map_and_select_item` was modified with the 3 `_list` commands below to reduce the number of boxes used in the computation.
This was the only way to run the program on a computer with only 16Gb RAM. These lines can be commented out to prevent this. Further development of the software could make this a paramete to add to the command line.

```{python}
    # Limit to first 70 boxes for testing
    box_list = box_list[:70]
    keep_list = keep_list[:70]
    total_list = total_list[:70]
    return np.asarray(box_list), np.asarray(keep_list), np.asarray(total_list)
```

The software as it is now was tested on a Silicon M1 macMini with 16 GB of RAM on the supplied `data` files with command:

```{bash}
python predict.py -p "data/6FBV.pdb" -e "data/emd_4230.map" -o "results" -m 3 
```

The size of files depending on `box_list` etc. parameters were as follows for various tested values. The "Max" values were deducted from the result `.pdb` file provided by the authors as `results/pdb_plus_cryoem/rmsf_nor.pdb`

| `.pth` file size. | Box Size | B-factor Range | Absolute Difference |
|-------------------|----------|----------------|---------------------|
| 4.8 Mb            | 10       | -1.42 to 1.80  | 3.22                |
| 24 Mb             | 50       | -1.19 to 3.61  | 4.80                |
| 39 Mb             | 70       | -1.01 to 4.54  | 5.55                |
| 191 Mb            | Max      | -0.82 to 6.95  | 7.77                |

## Web-Fronted RMSF-net hangs
As noted in [issue #3](https://github.com/XintSong/RMSF-net/issues/3) on the original repo, the "Web-Fronted RMSF-net software" just "hanged" on a Windows 11 laptop, probably for the lack of "exit" command at the end of the `chimera_exe.cmd` file that would have been created.
