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
####### setup ########
#######################

super_args = parse_args(ARGS, s)
args = parser(super_args, experiment="plan")

#######################
####### models ########
#######################

env = load_environment(args["dataset"])
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

gpt_epoch = args["gpt_epoch"]
gpt = Knet.load(joinpath(args["savepath"], "state_$gpt_epoch.jld2"))

prior = Knet.load(joinpath(args["savepath"], "prior_state_$gpt_epoch.jld2"))

gpt.padding_vector = normalize_joined_single(dataset, atype(zeros(Float32, gpt.transition_dim-1)))

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
    rollout = [deepcopy(env.state_vector())] # TODO: what does concatenate do here?
end

## previous (tokenized) transitions for conditioning transformer
context = []
mses = []

T = env.max_episode_steps

for t in 1:T
    # TODO: observation preprocess
    state = env.state_vector()

    if dataset.normalized_reward
        observation = normalize_states(dataset, observation) # TODO: implement normalize_states
    end

    if t % args["plan_freq"] == 1
        prefix = make_prefix(observation, transition_dim) # TODO: implement and index
        #TODO: implement beam with prior
        sequence = beam_with_prior(
            prior, gpt, prefix; 
            denormalize_rew=dataset.denormalize_reward,
            steps = args["horizon"],
            beam_width = args["beam_width"],
            n_expand = args["n_expand"],
            likelihood_weight = args["prob_weight"],
            prob_threshold = args["prob_threshold"],
            discount = discount
        )
    else
        sequence = sequence[2:end]
    end

    if (t == 1)
        first_value = denormalize_values(dataset, sequence[2,end]) # TODO: index check
        first_search_value = denormalize_values(dataset, sequence[2,1]) # TODO: index check
    end
    println(denormalize_values(dataset, sequence[2,end])) # TODO: index check
    
    ## [ transition_dim x horizon ] convert sampled tokens to continuous latentplan
    sequence_recon = sequence
    
    ## [ action_dim ] index into sampled latentplan to grab first action
    feature_dim = dataset.observation_dim
    action = extract_actions(sequence_recon, feature_dim, action_dim, t=0) # TODO: implement
    if dataset.normalized_raw
        action = denormalize_actions(dataset, action)
        sequence_recon = denormalize_joined(dataset, sequence_recon)
    end
    ## execute action in environment
    next_observation, reward, terminal, _ = env.step(action)

    # antmaze code omitted here

    ## update return
    total_reward += reward
    discount_return += reward .* discount ^ (t-1)
    score = env.get_normalized_score(total_reward)
    
    push!(rollout, deepcopy(state))
    context = update_context(observation, action, reward) #TODO: implement
    @printf("[ plan ] t: $t / $T | r: $reward | R: $total_reward | score: $score | time: | %s | %s | %s\n", args["dataset"], args["exp_name"], args["suffix"])

    # add viz
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
    "prediction_error" => mean(mses)
)
open(json_path, "w") do io
    JSON.print(io, json_data)
end


