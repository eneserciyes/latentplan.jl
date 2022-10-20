include("datasets/sequence.jl")
include("utils/setup.jl")

using .Sequence: SequenceDataset, normalize_joined_single
using .Setup: parser
using ArgParse: ArgParseSettings, @add_arg_table!, parse_args

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
        arg_type = Int
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
args = parser(super_args, experiment="train")

#######################
####### dataset #######
#######################

env_name = occursin("-v", args["dataset"]) ? args["dataset"] : args["dataset"] * "-v0"

# env params
sequence_length = args["subsampled_sequence_length"] * args["step"]
args["logbase"] = expanduser(args["logbase"])
args["savepath"] = expanduser(args["savepath"])
if !isdir(args["savepath"])
    mkdir(args["savepath"])
end

dataset = SequenceDataset(
    env_name;
    penalty=args["termination_penalty"], 
    sequence_length=sequence_length, 
    step=args["step"], 
    discount=args["discount"], 
    disable_goal=args["disable_goal"], 
    normalize_raw=args["normalize"], 
    normalize_reward=args["normalize_reward"],
    max_path_length=args["max_path_length"]
)

obs_dim = dataset.observation_dim
act_dim = dataset.action_dim
if args["task_type"] == "locomotion"
    transition_dim = obs_dim+act_dim+3
else
    transition_dim = 128+act_dim+3
end

#######################
######## model ########
#######################

# model = Model()
if args["normalize"]
    normalize_joined_single(dataset, zeros(25))
end



