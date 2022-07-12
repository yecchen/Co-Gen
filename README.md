
## Co-Gen

This is the code and data for the SIGIR'22 paper 
"Structured and Natural Responses Co-generation for Conversational Search".

### Requirements
- Python 3.6
- PyTorch 1.4.0

Other required modules are listed in requirements.txt, you could set up a conda environment for the experiment by:  
```bash
conda create -n Co-Gen python=3.6
conda activate Co-Gen
pip install -r requirements.txt 
```

### Folder
- configs: the hyperparameters for Co-Gen SL and RL training.
- data: all the needed training/validation/testing data for MultiWOZ 2.0 and MultiWOZ 2.1.
- latent_dialog: source code for Co-Gen model.
- reinforce.py: code for RL training and evaluation.
- supervised.py: code for SL training and evaluation.

### Data Preparation
The input data files are zipped put under directory ./data, it should be unzipped:
```
unzip data/act.zip -d data
unzip data/multiwoz_2.0.zip -d data
unzip data/multiwoz_2.1.zip -d data
```
You can also download MultiWOZ 2.0 & 2.1 data from the [official repository](https://github.com/budzianowski/multiwoz), and process it with its given delexicalization script. 

### Train a Co-Gen model
The training process contains two stage: Supervised Training and Reinforcement Learning.

SL training for MultiWOZ2.0 / 2.1:
```
python -u supervised.py --config_name sl_woz2.0
python -u supervised.py --config_name sl_woz2.1
```

RL training for MultiWOZ2.0 / 2.1:
```
python -u reinforce.py --config_name rl_woz2.0
python -u reinforce.py --config_name rl_woz2.1
```

To run evaluation on the trained SL/RL model:
```
python -u supervised.py --config_name sl_woz2.0 --forward_only
python -u supervised.py --config_name sl_woz2.1 --forward_only

python -u reinforce.py --config_name rl_woz2.0 --forward_only
python -u reinforce.py --config_name rl_woz2.1 --forward_only
```