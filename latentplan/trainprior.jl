using ArgParse: ArgParseSettings, @add_arg_table!, parse_args
using Statistics: mean
using Printf
using Knet
using Debugger: @enter, @bp, @run
using JSON

include("LPCore.jl")
include("setup.jl")
using .LPCore

s = ArgParseSettings()
@add_arg_table! s begin
    "--dataset"
        help = "which environment to use"
        arg_type = String
        default = "halfcheetah-medium-expert-v2"
    "--exp_name"
        help = "name of the experiment"
        arg_type = String
        default = "debug"
    "--seed"
        help = "seed"
        default = 42
    "--config"
        help = "relative jl file path with configurations"
        arg_type = String
        default = "../config/vqvae.jl"
end

#######################
######## setup ########
#######################

super_args = parse_args(ARGS, s)
args = parser(super_args, experiment="plan")

env_name = occursin("-v", args["dataset"]) ? args["dataset"] : args["dataset"] * "-v0"

args["logbase"] = expanduser(args["logbase"])
args["savepath"] = expanduser(args["savepath"])

sequence_length = args["subsampled_sequence_length"] * args["step"]

dataset = SequenceDataset(
    env_name;
    penalty=args["termination_penalty"], 
    sequence_length=sequence_length, 
    step=args["step"], 
    discount=args["discount"], 
    disable_goal=args["disable_goal"], 
    normalize_raw=args["normalize"], 
    normalize_reward=args["normalize_reward"],
    max_path_length=args["max_path_length"],
    atype=atype
)

obs_dim = dataset.observation_dim
act_dim = dataset.action_dim
transition_dim = dataset.joined_dim+1

gpt_epoch = args["gpt_epoch"]
representation = Knet.load(joinpath(args["savepath"], "state_$gpt_epoch.jld2"))
representation.padding_vector = normalize_joined_single(dataset, atype(zeros(Float32, representation.transition_dim-1)))

args = parser(super_args, experiment="train")
args["logbase"] = expanduser(args["logbase"])
args["savepath"] = expanduser(args["savepath"])
block_size = args.subsampled_sequence_length ÷ args.latent_step
obs_dim = dataset.observation_dim

model_config = deepcopy(args)
model_config["block_size"] = block_size
model_config["observation_dim"] = obs_dim
model_config["n_embd"] = args["n_embd"] * args["n_head"]

model = TransformerPrior(model_config)

warmup_tokens = length(dataset) * block_size
final_tokens = 20 * warmup_tokens

trainer_config = Dict(
    "batch_size" => args["batch_size"],
    "learning_rate" => args["learning_rate"],
    "betas" => (0.9, 0.95),
    "weight_decay" => 0.1,
    "grad_norm_clip" => 1.0,
    "warmup_tokens" => warmup_tokens,
    "kl_warmup_tokens" => warmup_tokens * 10,
    "final_tokens" => final_tokens,
    "lr_decay" => args["lr_decay"],
)

#######################
###### main loop ######
#######################

## scale number of epochs to keep number of updates constant
n_epochs = Int(floor(1e6 / length(dataset) * args["n_epochs_ref"]))
save_freq = Int(floor(n_epochs / args["n_saves"]))
#TODO: wandb init

for epoch in 1:n_epochs
    @printf("\nEpoch: %d / %d | %s | %s", epoch, n_epochs, env_name, args["exp_name"])

    save_epoch = (epoch + 1) ÷ save_freq * save_freq
    statepath = joinpath(args["savepath"], "prior_state_$save_epoch.jld2")

    #TODO: model save
end
