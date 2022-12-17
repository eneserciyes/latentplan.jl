# Efficient Planning in a Compact Latent Action Space

Unofficial Knet.jl implementation of the "[Efficient Planning in a Compact Latent Action Space](https://arxiv.org/abs/2208.10291)".

This version is being implemented by Enes Erciyes for the Koç University Comp 541 Course. You can find the original implementation in [https://github.com/ZhengyaoJiang/latentplan](https://github.com/ZhengyaoJiang/latentplan).


## Setting up D4RL and PyCall

* Download and place [MuJoCo 2.10](https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz) in `~/.mujoco/mujoco210`.
* Add this path to LD_LIBRARY_PATH in your shell init script. 
* Add `PyCall` and `Conda` in your Julia environment.
* Inside a Julia REPL, set `ENV["PYTHON"] = ""` and run `using PyCall`. This will set up a conda environment called `conda_jl`. 
* Inside a Julia REPL, run `Conda.add("mesalib"; channel="conda-forge")`
* Then, run `Conda.pip_interop(true)` to be able to install pip dependencies.
* Install the pip dependencies using:
```
Conda.pip("install", ["mujoco-py==2.1.2.14", "git+https://github.com/JannerM/d4rl.git@c3dd04da02acbf4de6cbaa1141deb4f958f03ca9", "dm_control", "git+https://github.com/aravindr93/mjrl@3871d93763d3b49c4741e6daeaebbc605fe140dc"])
```
* Outside the Julia REPL, run
```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/conda/pkgs/[mesalib-pkg-dir]/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/conda/pkgs/[zstd-pkg-dir]/lib
```
* Check if the setup is successful by importing `d4rl`:
```julia
using PyCall
d4rl = pyimport("d4rl")
```
NOTE: Flow and CARLA environments are not used here, therefore they are not included in the setup.
