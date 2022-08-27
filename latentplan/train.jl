using Images, Interpolations
using Images.FileIO
using Wandb

using FileIO
using CUDA: CUDA, CuArray
using Knet

println(CUDA.functional())
