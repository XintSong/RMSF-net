CONDA_PATH=$(dirname $(dirname $(which conda)))
conda create --name rmsf-net python=3.8
source $CONDA_PATH/bin/activate rmsf-net
pip install -r requirements.txt
conda install moleculekit  -c acellera -c conda-forge

