# RLBackgammon
## Usage
### 1. Create config file (_.yaml_).  
Required arguments:
   * _**algo_name**_ - name of the algorithm from [Stable Baselines](https://github.com/DLR-RM/stable-baselines3)
   * _**discr_space**_ - ability of the model to act in discrete space
   * _**start_pos**_ - Starting position for algorithm to generate the field.
   Possible values: ['Short', 'LongGammon']  
   * _**timesteps**_ - amount of timestamps for model to be trained  

Optional arguments
   * _**algo_args**_ - argument to be passed into model. If not given, default will be used.
   Example:
```yaml
model:
    - DQN:
      algo_name:   "DQN"
      discr_space: True
      start_pos:   "LongGammon"
      timesteps:   100000
      algo_args:
        gamma:                0.99
        learning_rate:        0.001
        exploration_fraction: 0.9
        batch_size:           32
    - DDPG:
      algo_name:   "DDPG"
      discr_space: False
      start_pos:   "LongGammon"
      timesteps:   100000
      algo_args:
        gamma:                0.99
        learning_rate:        0.001
```

### 2. Train models
run
```bash
python train.py
```
After completion, models will be saved in _./models_ directory.
### 3. Run simulations
run
```bash
python play.py
```
Result table will be saved in _result.csv_ file in the project root directory.  
File will be containing winrates between each trained model in one-by-one game simulation.  
_RandomAgent_ is also added to the simulation list to measure models performance against random way of playing a game.
