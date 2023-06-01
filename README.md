# Distributionally Robust Policy Gradient for Offline Contextual Bandits

This repository provides python implementation for our paper [Distributionally Robust Policy Gradient for Offline Contextual Bandits](https://proceedings.mlr.press/v206/yang23f.html) to appear in ATSTATS 2023.

### Abstract
Learning an optimal policy from offline data is notoriously challenging, which requires the evaluation of the learning policy using data pre-collected from a static logging policy. We study the policy optimization problem in offline contextual bandits using policy gradient methods. We employ a distributionally robust policy gradient method, DROPO, to account for the distributional shift between the static logging policy and the learning policy in policy gradient. Our approach conservatively estimates the conditional reward distributional and updates the policy accordingly. We show that our algorithm converges to a stationary point with rate 𝑂(1/𝑇)
, where 𝑇 is the number of time steps. We conduct experiments on real-world datasets under various scenarios of logging policies to compare our proposed algorithm with baseline methods in offline contextual bandits. We also propose a variant of our algorithm, DROPO-exp, to further improve the performance when a limited amount of online interaction is allowed. Our results demonstrate the effectiveness and robustness of the proposed algorithms, especially under heavily biased offline data.
 
### Experiments
The code for DROPO is in `DROPO/` folder and the code for DROPO-exp is in  `DROPO-exp/` folder. 

#### DROPO
First, we go the the `DROPO/` folder.
```console
$ cd DROPO
```
We run the simulation with the following commands:
```console
# baseline methods
$ python baseline_simulation.py --mode 1 # DM
$ python baseline_simulation.py --mode 2 # IPS
$ python baseline_simulation.py --mode 3 # DR

$ python DROPO_simulation.py # DROPO
```

#### DROPO-exp

```console
$ cd DROPO-exp
```
We run the simulation with the following commands:
```console
# baseline methods
$ python baseline_exp_simulation.py --mode 1 # DM
$ python baseline_exp_simulation.py --mode 2 # IPS
$ python baseline_exp_simulation.py --mode 3 # DR

$ python DROPO_exp_simulation.py # DROPO
```


