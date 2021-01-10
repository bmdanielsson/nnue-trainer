# Overview
This repository contains the code used to train NNUE style networks for the Marvin chess engine.

# Setup
```
python3 -m venv env
source env/bin/activate
pip install python-chess torch
```

# Building the data loader
This requires a C++17 compiler.

Windows:
```
compile_data_loader.bat
```

Linux/Mac:
```
sh compile_data_loader.bat
```

# Basic training steps
```
convert training.bin training.binpack
convert validation.bin validation.binpack
python train.py training.binpack validation.binpack
python serialize.py last.pt nn.nnue
```

# Acknowledgements
* Data loader and a lot of other code is taken from https://github.com/glinscott/nnue-pytorch
* Ranger optimizer is taken from https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer

