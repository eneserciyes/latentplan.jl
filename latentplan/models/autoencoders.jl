module AutoEncoders

include("common.jl")

using .Common: Chain, Linear, ReLU

export encode, decode, MLPModel, SymbolWiseTransformer, StepWiseTransformer

struct Encoder;
    MLP;
    linear_means;
    linear_log_var;
    
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
end


function (e::Encoder)(x)
    x = e.MLP(x)
    means = e.linear_means(x)
    log_vars = e.linear_log_var(x)
    return means, log_vars
end


struct Decoder; 
    MLP; 

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
end

function (d::Decoder)(z)
    x = d.MLP(z)
    return x
end


struct MLPModel
    condition_size::Int32;
    trajectory_input_length::Int32;
    encoder::Encoder;
    decoder::Decoder;

    function MLPModel(config)
    end
end

function encode(m::MLPModel, X)
end

function decode(m::MLPModel, latents, state)
end


struct SymbolWiseTransformer
    latent_size;
    condition_size;
    trajectory_input_length;
    embedding_dim;
    trajectory_length;
    block_size;
    observation_dim;
    action_dim;
    transition_dim;
    encoder;
    decoder;
    pos_emb;
    state_emb;
    action_emb;
    reward_emb;
    value_emb;
    pred_state;
    pred_action;
    pred_reward;
    pred_value;
    linear_means;
    linear_log_var;
    latent_mixing;
    ln_f;
    drop;

    function SymbolWiseTransformer(config)
    end
end

function encode(sym_t::SymbolWiseTransformer, joined_inputs)
end

function decode(sym_t::SymbolWiseTransformer, latents, state)
end

struct StepWiseTransformer
    latent_size;
    condition_size;
    trajectory_input_length;
    embedding_dim;
    trajectory_length;
    block_size;
    observation_dim;
    action_dim;
    transition_dim;
    encoder;
    decoder;
    pos_emb;
    embed;
    predict;
    linear_means;
    linear_log_var;
    latent_mixing;
    ln_f;
    drop;
end

function encode(step_t::StepWiseTransformer, joined_inputs)
end

function decode(step_t::StepWiseTransformer, latents, state)
end

end
