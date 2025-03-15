# Leatherback Tutorial for Isaac Lab Study Group
```
Credit: 
Strainflow (Eric)
Antoine: https://github.com/AntoineRichard/
Papachuck: https://github.com/renanmb

Maintainer:
Genozen https://github.com/Genozen

Licensing: BSD-3-Clause

03/14/2025
```


# Commands:
Training:
```
# for training
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py --task Leatherback_1-Direct-v0 --headless
```
Post-training:
```
# after training
./isaaclab.sh -p scripts/reinforcement_learning/skrl/play.py --task Leatherback_1-Direct-v0 --num_envs 32
```
