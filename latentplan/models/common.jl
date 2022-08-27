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
end
