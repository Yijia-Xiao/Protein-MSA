"""scp -r /dataset/ee84df8b/RaptorX-3DModeling/ ~/
scp -r /dataset/ee84df8b/database/ ~/
conda install -c conda-forge -c bioconda hhsuite -y"""

cd ~/RaptorX-3DModeling/
echo "export HHDIR=/opt/conda/
export PATH=$HHDIR/bin:$HHDIR/scripts:$PATH
export HHDB=/root/database/uniclust30_2018_08


export ModelingHome=/root/RaptorX-3DModeling
export DistFeatureHome=$ModelingHome/BuildFeatures/
export DL4DistancePredHome=$ModelingHome/DL4DistancePrediction4/
export DL4PropertyPredHome=$ModelingHome/DL4PropertyPrediction/
export DistanceFoldingHome=$ModelingHome/Folding/
export DistFeatureHome=$ModelingHome/BuildFeatures/
export PYTHONPATH=$ModelingHome:$PYTHONPATH
export PATH=$ModelingHome/bin:$PATH" >> ~/.bashrc

source ~/.bashrc
export HHDIR=/opt/conda/
export PATH=$HHDIR/bin:$HHDIR/scripts:$PATH
export HHDB=/root/database/uniclust30_2018_08

export ModelingHome=/root/RaptorX-3DModeling
export DistFeatureHome=$ModelingHome/BuildFeatures/
export DL4DistancePredHome=$ModelingHome/DL4DistancePrediction4/
export DL4PropertyPredHome=$ModelingHome/DL4PropertyPrediction/
export DistanceFoldingHome=$ModelingHome/Folding/
export DistFeatureHome=$ModelingHome/BuildFeatures/
export PYTHONPATH=$ModelingHome:$PYTHONPATH
export PATH=$ModelingHome/bin:$PATH

python ft.py $id
