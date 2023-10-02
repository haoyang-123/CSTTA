# CSTTA

## Prerequisite
To reproduce our results, please kindly create and use this environment.
```
# It may take several minutes for conda to solve the environment
conda update conda
conda env create -f environment.yml
conda activate cstta 
```

## Experiment 

### ImageNet-to-ImageNetC task 
```bash
# Tested on RTX3090
python -u imagenetc.py --cfg cfgs/cstta.yaml
```


