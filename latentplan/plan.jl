using ArgParse: ArgParseSettings, @add_arg_table!, parse_args
using Statistics: mean
using Printf
using Knet
using Debugger: @enter, @bp, @run
using JSON

include("LPCore.jl")
include("setup.jl")

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
    "--beam_width"
        default = 64
    "--n_expand"
        default = 4
    "--suffix"
        default = ""
    "--config"
        help = "relative jl file path with configurations"
        arg_type = String
        default = "../config/vqvae.jl"
end

#######################
####### util functions ########
#######################

function make_prefix(obs, transition_dim)
    obs_discrete = atype(obs)
    pad_dims = atype(zeros(transition_dim - size(obs_discrete, 1)))
    if ndims(obs_discrete) == 2
        obs_discrete = reshape(obs_discrete, :, 1, 1)
        pad_dims = reshape(pad_dims, :, 1, 1)
    end
    transition = cat(obs_discrete, pad_dims, dims=1)
    prefix = transition
    return prefix
end

function extract_actions(x, observation_dim, action_dim; t=nothing)
    actions =  x[observation_dim+1:observation_dim+action_dim, :]
    if t !== nothing
        return actions[:, t]
    else
        return actions
    end
end

VALUE_PLACEHOLDER = 1e6
function update_context(observation, action, reward)
    rew_val = [reward; VALUE_PLACEHOLDER]
    transition = cat(observation, action, rew_val; dims=1)
    context = []
    transition_discrete = atype(transition)
    transition_discrete = reshape(transition_discrete, :, 1, 1)
    push!(context, transition_discrete)
    return context
end

#######################
####### setup ########
#######################

super_args = parse_args(ARGS, s)
args = parser(super_args, experiment="plan")

args["logbase"] = expanduser(args["logbase"])
args["savepath"] = expanduser(args["savepath"])
args["loadpath"] = joinpath(args["logbase"], args["dataset"], args["exp_name"])

#######################
####### models ########
#######################

env = load_environment(args["dataset"])
dataset_config = Knet.load(joinpath(args["loadpath"] , "dataset_config.jld2"), "config")

dataset = SequenceDataset(
    dataset_config["env_name"];
    penalty=dataset_config["penalty"],
    sequence_length=dataset_config["sequence_length"], 
    step=dataset_config["step"], 
    discount=dataset_config["discount"], 
    disable_goal=dataset_config["disable_goal"], 
    normalize_raw=dataset_config["normalize_raw"], 
    normalize_reward=dataset_config["normalize_reward"],
    max_path_length=dataset_config["max_path_length"],
    atype=dataset_config["atype"]
)

gpt_epoch = args["gpt_epoch"]

gpt = Knet.load(joinpath(args["loadpath"], "state_$gpt_epoch.jld2"))
prior = Knet.load(joinpath(args["loadpath"], "prior_state_$gpt_epoch.jld2"))

discount = dataset.discount
observation_dim = dataset.observation_dim
action_dim = dataset.action_dim

#######################
###### main loop ######
#######################
REWARD_DIM = VALUE_DIM = 1
transition_dim = observation_dim + action_dim + REWARD_DIM + VALUE_DIM

observation = env.reset()
total_reward = 0
discount_return = 0

if occursin("antmaze", env.name)
    if dataset.disable_goal
        observation = cat(observation, atype(zeros(Float32, 2)); dims=1)
        rollout = [cat(deepcopy(env.state_vector()), atype(zeros(Float32, 2)); dims=1)]
    else
        observation = cat(observation, env.target_goal; dims=1)
        rollout = [cat(deepcopy(env.state_vector()), env.target_goal; dims=1)]
    end
else
    rollout = [deepcopy(env.state_vector())]
end

## previous (tokenized) transitions for conditioning transformer
context = []
mses = []

T = env.max_episode_steps

for t in 1:T
    state = env.state_vector()

    if dataset.normalized_reward
        observation = normalize_states(dataset, observation)
    end

    if t % args["plan_freq"] == 1
        prefix = make_prefix(observation, transition_dim)
        sequence = beam_with_prior(
            prior, gpt, prefix, dataset,
            discount = discount,
            steps = args["horizon"],
            beam_width = args["beam_width"],
            n_expand = args["n_expand"],
            likelihood_weight = args["prob_weight"],
            prob_threshold = args["prob_threshold"],
        ) # [17 x 3]
    else
        sequence = sequence[:, 2:end]
    end

    if (t == 1)
        first_value = denormalize_values(dataset, sequence[end-1, 1])
        first_search_value = denormalize_values(dataset, sequence[end-1,end])
    end
    println(denormalize_values(dataset, sequence[end-1, 1]))
    
    ## [ transition_dim x horizon ] convert sampled tokens to continuous latentplan
    sequence_recon = sequence
    
    ## [ action_dim ] index into sampled latentplan to grab first action
    feature_dim = dataset.observation_dim
    action = extract_actions(sequence_recon, feature_dim, action_dim; t=0) 
    if dataset.normalized_raw
        action = denormalize_actions(dataset, action)
        sequence_recon = denormalize_joined(dataset, sequence_recon)
    end
    ## execute action in environment
    next_observation, reward, terminal, _ = env.step(action)

    # antmaze code omitted here

    ## update return
    total_reward += reward
    discount_return += reward * (discount ^ (t-1))
    score = env.get_normalized_score(total_reward)
    
    push!(rollout, deepcopy(state))
    context = update_context(observation, action, reward)
    @printf("[ plan ] t: %d / %d | r: %.2f | R: %.2f | score: %.4f | time: | %s | %s | %s\n", t, T, reward, total_reward, score, args["dataset"], args["exp_name"], args["suffix"])

    # TODO: add viz
    if terminal
        break
    end
    observation = next_observation
end

# save result as a json file
json_path = joinpath(args["savepath"], "rollout.json")
json_data = Dict(
    "score" => score,
    "step" => t,
    "return" => total_reward,
    "term" => terminal,
    "gpt_epoch" => gpt_epoch,
    "first_value" => first_value,
    "first_search_value" => first_search_value,
    "discount_return" => discount_return,
    # "prediction_error" => mean(mses)
)
open(json_path, "w") do io
    JSON.print(io, json_data)
end


