# Overview
This repository contains the code used to train NNUE style networks for the Marvin chess engine.

# Setup
```
python3 -m venv env
source env/bin/activate
pip install python-chess torch tensorboard
```

# Building the data loader
This requires the Clang compiler.

```
make
```

# Basic training steps
```
python train.py training.bin validation.bin
```

# Logging
```
tensorboard --logdir=logs
```
Then, go to http://localhost:6006/

# Acknowledgements
* Training code is based on https://github.com/glinscott/nnue-pytorch
