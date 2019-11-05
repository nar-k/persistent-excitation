# persistent-excitation
The implementation of the experiments in "Persistency of Excitation for Robustness of Neural Networks" by Kamil Nar and S. Shankar Sastry. The paper is available at https://arxiv.org/abs/1911.01043.  

Run the following commands in command-line for each figure while in the directory of the repository.

### Figure 1
```
python ce-vs-sq.py --seed 1
python ce-vs-sq.py --seed 21
python ce-vs-sq.py --seed 41
python ce-vs-sq.py --seed 61
python ce-vs-sq.py --seed 81
python ce-vs-sq.py --seed 101
```
### Figure 2
```
python poor-margins-of-ce.py
```
### Figure 4
```
python train-ce.py
python train-persist.py --perturb-mag 0
```
### Figure 5
```
python train-persist.py --perturb-mag 0.005
python train-persist.py --perturb-mag 0.010
python train-persist.py --perturb-mag 0.020
```
### Figure 6
```
python train-persist.py --perturb-mag 0.020 --perturb-only-first-layer False
python train-persist.py --perturb-mag 0.020 --perturb-only-first-layer True
```
