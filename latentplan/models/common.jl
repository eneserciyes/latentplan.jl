module Common

using Knet: relu

export Chain
struct Chain
    layers
    Chain(layers...) = new(layers)
end
(c::Chain)(x) = (for l in c.layers; x = l(x); end; x)

export Linear
struct Linear; w; b; end
Linear(in_dim::Int, out_dim::Int) = Linear(param(out_dim,in_dim,atype=atype), param0(out_dim,atype=atype))
(l::Linear)(x) = l.w * x .+ l.b
paramlist(l::Linear) = [l.w, l.b]


export ReLU
struct ReLU; end
(r::ReLU)(x) = relu.(x)

export Embedding
struct Embedding
    weight::Param{Matrix{Float32}}

    function Embedding(D, K)
        weight = Param(rand(Uniform(-1/K, 1/K), (D, K)))
        new(weight)
    end
end

function (e::Embedding)(x)
    weight * transpose(x)
end

export one_hot
function one_hot(Type, indices, class_num)
    onehot = zeros(Type, class_num, size(indices)...)
    for index in CartesianIndices(indices)
        onehot[indices[index], index] = convert(Type, 1)
    end
    onehot
end



