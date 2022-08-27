include("common.jl")

using .Common: Chain, Linear, ReLU

struct Encoder;
    MLP;
    linear_means;
    linear_log_var;
end

function Encoder(layer_sizes::Array, latent_size::Int, condition_size::Int)
    MLP = Chain()
    layer_sizes[0] += condition_size

    in_sizes = layer_sizes[1:end-1]
    out_sizes = layer_sizes[2:end]
    layer_count = size(in_sizes)
    for i = 1:layer_count
        push!(MLP.layers, Linear(in_sizes[i],out_sizes[i]))
        push!(MLP.layers, ReLU())
    end
    linear_means = Linear(out_sizes[end], latent_size)
    linear_log_var = Linear(out_sizes[end], latent_size)
    new(MLP, linear_means, linear_log_var)
end

function (e::Encoder)(x)
    x = e.MLP(x)
    means = e.linear_means(x)
    log_vars = e.linear_log_var(x)
    return means, log_vars
end


struct Decoder; MLP; end

function Decoder(layer_sizes::Array, latent_size::Int, condition_size::Int)
    MLP = Chain()
    input_size = latent_size + condition_size

    in_sizes = push(input_size, layer_sizes[1:end-1])
    out_sizes = layer_sizes
    layer_count = size(in_sizes)
    for i = 1:layer_count
        push!(MLP.layers, Linear(in_sizes[i],out_sizes[i]))
        push!(MLP.layers, ReLU())
    end
end
