module LPCore

using Knet
using CUDA: functional as cuda_available

export atype
if cuda_available()
    atype=KnetArray{Float32}
else
    atype=Array{Float32}
end

include("datasets/sequence.jl")
include("models/common.jl")
include("models/transformers.jl")
include("models/vqvae.jl")

end
