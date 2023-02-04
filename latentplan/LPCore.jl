using Knet
using CUDA: CuArray, functional as cuda_available

export atype
if cuda_available()
    atype=CuArray{Float32}
else
    atype=Array{Float32}
end
cputype=Array{Float32}

include("datasets/sequence.jl")
include("models/common.jl")
include("models/transformers.jl")
include("models/vqvae.jl")
