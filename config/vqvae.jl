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

logbase = "~/logs_julia/"
gpt_expname = "vae/vq"

args_to_watch = [
    ("prefix", ""),
    ("plan_freq", "freq"),
    ("horizon", "H"),
    ("beam_width", "beam"),
]

export Train
@option "train_base" struct Train <: AbstractConfig
    model= "VQTransformer"
    tag= "experiment"
    state_conditional = true
    N = 100
    discount = 0.99
    n_layer = 4
    n_head = 4

    ## number of epochs for a 1M-size dataset; n_epochs = 1M / dataset_size * n_epochs_ref
    n_epochs_ref = 50
    n_saves = 3
    logbase= logbase
    device= "cuda"

    K = 512
    latent_step = 3
    n_embd = 128
    trajectory_embd = 512
    batch_size = 512
    learning_rate = 2e-4
    lr_decay = false
    seed = 42

    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    step = 1
    subsampled_sequence_length = 25
    termination_penalty = -100
    exp_name= gpt_expname

    position_weight = 1
    action_weight = 5
    reward_weight = 1
    value_weight = 1

    first_action_weight = 0
    sum_reward_weight = 0
    last_value_weight = 0
    suffix= ""

    normalize = true
    normalize_reward = true
    max_path_length = 1000
    bottleneck= "pooling"
    masking= "uniform"
    disable_goal = false
    residual = true
    ma_update = true

end

export Plan
@option "plan_base" struct Plan <: AbstractConfig
    discrete = false
    logbase= logbase
    gpt_loadpath= gpt_expname
    gpt_epoch= "latest"
    device= "cuda"
    renderer= "Renderer"
    suffix= "0"

    plan_freq = 1
    horizon = 15
    iql_value = false

    rounds = 2
    nb_samples = 4096

    beam_width = 32
    n_expand = 4

    prob_threshold = 0.05
    prob_weight = 5e2

    vis_freq = 200
    exp_name::Function = watch(args_to_watch)
    verbose = true
    uniform = false

    # Planner
    test_planner= "beam_prior"
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
        "subsampled_sequence_length" => 25,
        "n_layer" => 3,
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
