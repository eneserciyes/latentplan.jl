# Efficient Planning in a Compact Latent Action Space

Unofficial Knet.jl implementation of the "[Efficient Planning in a Compact Latent Action Space](https://arxiv.org/abs/2208.10291)".

This version is being implemented by Enes Erciyes for the Koç University Comp 541 Course. You can find the original implementation in [https://github.com/ZhengyaoJiang/latentplan](https://github.com/ZhengyaoJiang/latentplan).


## Setting up MuJoCo

* Download and place [MuJoCo 2.10](https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz) in `~/.mujoco/mujoco210`.
* Add this path to LD_LIBRARY_PATH in your shell init script. 
* Install Julia MuJoCo wrapper
```
pkg> registry add https://github.com/Lyceum/LyceumRegistry.git
add MuJoCo
```
* Download a MuJoCo license from [http://www.roboti.us/license.html](http://www.roboti.us/license.html)

* Activate the Lyceum wrapper (MuJoCo no longer requires a license but the wrapper is unmaintained and still requires it.)

```julia
using MuJoCo
mj_activate("mjkey.txt")
```

* Test if the setup works
```julia
m = jlModel("humanoid.xml")
d = jlData(m)
for i=1:100
    mj_step(m, d);
    println(d.qpos)
end
```

