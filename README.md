# drl
Deep reinforcement leanring algorithms implemented in PyTorch.

Refactored [ShangtongZhang's DeepRL](https://github.com/ShangtongZhang/DeepRL) to ease the running and do some optimization. All rights reserved to him.

My contributions are:
* New implementations of PPO and VPG according to [spinningup](https://spinningup.openai.com/en/latest/).
* Optimize the replay buffer of DQN to reduce the memory footprint (from 16GB memory requiremnt to 1.8GB).
