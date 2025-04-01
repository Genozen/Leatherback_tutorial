# Leatherback Tutorial for Isaac Lab Study Group
```
Credit: 
Strainflow (Eric)
Antoine Richard: https://github.com/AntoineRichard/
Papachuck: https://github.com/renanmb

Maintainer:
Genozen https://github.com/Genozen

Licensing: BSD-3-Clause

03/14/2025
```


## Documentation
https://lycheeai.notion.site/Leatherback-Community-Project-1b828763942b818c903be4648e53f23d?pvs=4

Specifications: </br>
`Isaac Sim = 4.5.0` </br>
`Isaac Lab = 2.0` </br>
`CUDA = 12.` </br>
`Linux/Windows` </br>


## Quick Commands
Linux
```
# training
python scripts/reinforcement_learning/skrl/train.py --task Isaac-Leatherback-Direct-v0 --num_envs 32
python scripts/reinforcement_learning/skrl/train.py --task Isaac-Leatherback-Direct-v0 --num_envs 4096 --headless

# playing
python scripts/reinforcement_learning/skrl/play.py --task Isaac-Leatherback-Direct-v0 --num_envs 32
```

Windows
```
# training
python scripts\reinforcement_learning\skrl\train.py --task Isaac-Leatherback-Direct-v0 --num_envs 32
python scripts\reinforcement_learning\skrl\train.py --task Isaac-Leatherback-Direct-v0 --num_envs 4096 --headless

# playing
python scripts\reinforcement_learning\skrl\play.py --task Isaac-Leatherback-Direct-v0 --num_envs 32
```
