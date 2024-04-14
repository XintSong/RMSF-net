conda activate base
conda create --name rmsf-net python=3.8
pip install -r requirements.txt
conda install moleculekit  -c acellera -c conda-forge
