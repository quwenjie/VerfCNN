# VerfCNN

This repository contains the implementation of the paper **VerfCNN, Optimal Complexity zkSNARK for Convolutional Neural Networks**, IEEE S&P 2026.

Paper: https://eprint.iacr.org/2025/2020

---



VerfCNN relies on the `mcl` library.

```bash
sudo apt install libgmp-dev
git clone https://github.com/herumi/mcl
cd mcl
make -j4
sudo make install
```

Installing `pytorch` and `numpy` are also required for loading the dataset and running the model.

---
The following script loads the CNN model, performs inference, and generates the corresponding zkSNARK proof.
```bash
g++ -Wall -fPIC -c -I/usr/local/include convnet_params.cpp convnet.cpp tools.cpp hyrax.cpp logup.cpp -mcmodel=large -lmcl -lgmp -O2
g++ -shared convnet_params.o convnet.o tools.o hyrax.o logup.o -L/usr/local/lib -lmcl -lgmp -o convnet.so -O2
python3 main.py
```

