module VQVAEConfig

using Configurations: @option, from_dict

function watch(args_to_watch::Vector{Tuple{String, String}})::Function
    function _fn(args)
        exp_name = []
        for (key, label) in args_to_watch
            if !hasproperty(args, key)
                continue
            end
            
            val = getfield(args, key)
            push!(exp_name, "$label$val")
        end
        exp_name = join(exp_name, "_")
        exp_name = replace(exp_name, "/_" => "/")
        
    end
    return _fn
end

abstract type AbstractConfig end

logbase = "~/logs/"
gpt_expname = "vae/vq"

args_to_watch = [
    ("prefix", ""),
    ("plan_freq", "freq"),
    ("horizon", "H"),
    ("beam_width", "beam"),
]

export Train
@option "train_base" struct Train <: AbstractConfig
    model::String = "VQTransformer"
    tag::String = "experiment"
    state_conditional::Bool = true
    N::Int32 = 100
    discount::Float32 = 0.99
    n_layer::Int32 = 4
    n_head::Int32 = 4

    ## number of epochs for a 1M-size dataset; n_epochs = 1M / dataset_size * n_epochs_ref
    n_epochs_ref::Int32 = 50
    n_saves::Int32 = 3
    logbase::String = logbase
    device::String = "cuda"

    K::Int32 = 512
    latent_step::Int32 = 3
    n_embd::Int32 = 128
    trajectory_embd::Int32 = 512
    batch_size::Int32 = 512
    learning_rate::Float32 = 2e-4
    lr_decay::Bool = false
    seed::Int32 = 42

    embd_pdrop::Float32 = 0.1
    resid_pdrop::Float32 = 0.1
    attn_pdrop::Float32 = 0.1

    step::Int32 = 1
    subsampled_sequence_length::Int32 = 25
    termination_penalty::Union{Int32,Nothing} = -100
    exp_name::String = gpt_expname

    position_weight::Int32 = 1
    action_weight::Int32 = 5
    reward_weight::Int32 = 1
    value_weight::Int32 = 1

    first_action_weight::Int32 = 0
    sum_reward_weight::Int32 = 0
    last_value_weight::Int32 = 0
    suffix::String = ""

    normalize::Bool = true
    normalize_reward::Bool = true
    max_path_length::Int32 = 1000
    bottleneck::String = "pooling"
    masking::String = "uniform"
    disable_goal::Bool = false
    residual::Bool = true
    ma_update::Bool = true

end

export Plan
@option "plan_base" struct Plan <: AbstractConfig
    discrete::Bool = false
    logbase::String = logbase
    gpt_loadpath::String = gpt_expname
    gpt_epoch::String = "latest"
    device::String = "cuda"
    renderer::String = "Renderer"
    suffix::String = "0"

    plan_freq::Int32 = 1
    horizon::Int32 = 15
    iql_value::Bool = false

    rounds::Int32 = 2
    nb_samples::Int32 = 4096

    beam_width::Int32 = 32
    n_expand::Int32 = 4

    prob_threshold::Float32 = 0.05
    prob_weight::Float32 = 5e2

    vis_freq::Int32 = 200
    exp_name::Function = watch(args_to_watch)
    verbose::Bool = true
    uniform::Bool = false

    # Planner
    test_planner::String = "beam_prior"
end


export hammer_cloned_v0_train, human_expert_v0, hammer_human_v0
hammer_human_v0 = human_expert_v0 = hammer_cloned_v0 = Dict{String,AbstractConfig}(
    "train" => from_dict(Train, Dict{String,Any}(
        "termination_penalty" => nothing,
        "max_path_length" => 200,
        "n_epochs_ref" => 10,
        "subsampled_sequence_length" => 25
    )),
    "plan" => from_dict(Plan, Dict{String,Any}(
        "horizon" => 24,
    ))
)

export relocate_cloned_v0, relocate_human_v0, relocate_expert_v0
relocate_cloned_v0 = relocate_human_v0 = relocate_expert_v0 = hammer_cloned_v0

export door_cloned_v0, door_human_v0, door_expert_v0
door_cloned_v0 = door_human_v0 = door_expert_v0 = hammer_cloned_v0

export pen_cloned_v0, pen_expert_v0, pen_human_v0
pen_cloned_v0 = pen_expert_v0 = pen_human_v0 = Dict{String,AbstractConfig}(
    "train" => from_dict(Train, Dict{String,Any}(
        "termination_penalty" => nothing,
        "max_path_length" => 100,
        "n_epochs_ref" => 10,
        "subsampled_sequence_length" => 25
    )),
    "plan" => from_dict(Plan, Dict{String,Any}(
        "horizon" => 24,
        "prob_weight" => 5e2,
    ))
)

export antmaze_large_diverse_v0, antmaze_large_play_v0, antmaze_medium_diverse_v0, antmaze_medium_play_v0, antmaze_umaze_v0
antmaze_large_diverse_v0 = antmaze_large_play_v0 = antmaze_medium_diverse_v0 = antmaze_medium_play_v0 = antmaze_umaze_v0 = Dict{String,AbstractConfig}(
    "train" => from_dict(Train, Dict{String,Any}(
        "disable_goal" => false,
        "termination_penalty" => nothing,
        "max_path_length" => 1001,
        "normalize" => false,
        "normalize_reward" => false,
        "lr_decay" => false,
        "K" => 8192,
        "discount" => 0.998,
        "subsampled_sequence_length" => 16
    )),
    "plan" => from_dict(Plan, Dict{String,Any}(
        "iql_value" => false,
        "horizon" => 15,
        "vis_freq" => 200,
        "renderer" => "AntMazeRenderer",
    ))
)

export antmaze_ultra_diverse_v0, antmaze_ultra_play_v0
antmaze_ultra_diverse_v0=antmaze_ultra_play_v0 = Dict{String,AbstractConfig}(
    "train" => from_dict(Train, Dict{String,Any}(
        "disable_goal" => false,
        "termination_penalty" => nothing,
        "max_path_length" => 1001,
        "normalize" => false,
        "normalize_reward" => false,
        "lr_decay" => false,
        "K" => 8192,
        "discount" => 0.998,
        "batch_size" => 512,
        "subsampled_sequence_length" => 16
    )),
    "plan" => from_dict(Plan, Dict{String,Any}(
        "iql_value" => false,
        "horizon" => 15,
        "vis_freq" => 200,
        "renderer" => "AntMazeRenderer",
    ))
)


end
