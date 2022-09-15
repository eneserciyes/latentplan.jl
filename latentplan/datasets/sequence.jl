module Sequence
include("d4rl.jl")
include("../vector_utils.jl")

using .D4RL
using .Vutils: squeeze
using Printf
using Statistics: mean, std


function segment(observations, terminals, max_path_length)
    @assert size(observations, 2) == size(terminals, 1)
    observation_dim = size(observations, 1)
    trajectories = [[]]

    for (obs, term) in zip(eachcol(observations), terminals)
        push!(trajectories[end], obs)
        if term
            push!(trajectories, [])
        end 
    end

    if length(trajectories[end]) == 0
        trajectories = trajectories[:end-1]
    end

    trajectories = [reduce(hcat, trajectory) for trajectory in trajectories]
    n_trajectories = length(trajectories)
    path_lengths = [size(traj, 2) for traj in trajectories]

    # pad trajectories to be equal length
    trajectories_pad = zeros(eltype(trajectories[1]), (observation_dim, max_path_length, n_trajectories))
    early_termination = zeros(Bool, (max_path_length, n_trajectories))
    for (i, traj) in enumerate(trajectories)
        path_length = path_lengths[i]
        trajectories_pad[:, 1:path_length, i] = traj
        early_termination[path_length:end, i] .= 1
    end
    return trajectories_pad, early_termination, path_lengths
end 

struct SequenceDataset;
    env;
    sequence_length;
    step;
    max_path_length;
    device;
    disable_goal;
    normalized_raw;
    normalize_reward;
    obs_mean; obs_std;
    act_mean; act_std;
    reward_mean; reward_std;
    # observations_raw;
    # actions_raw;
    # joined_raw;
    # rewards_raw;
    # terminals_raw;
    # joined_segmented; termin
    function SequenceDataset(env; sequence_length::Int=250, step::Int=10, 
        discount::Float64=0.99, max_path_length::Int=1000,
        penalty=nothing, device::String="cuda:0", normalize_raw::Bool=true, normalize_reward::Bool=true,
        train_portion::Float64=1.0, disable_goal::Bool=false)
    
        @printf("[ datasets/sequence ] Sequence length: %d | Step: %d | Max path length: %d\n", sequence_length, step, max_path_length)
        
        env = typeof(env) == String ? load_environment(env) : env
        println("[ datasets/sequence ] Loading...")

        dataset = qlearning_dataset_with_timeouts(env.unwrapped, terminate_on_end=true, disable_goal=disable_goal, debug=true)
        print('✓')

        # TODO: preprocess_fn

        ##
        observations = dataset["observations"]
        actions = dataset["actions"]
        rewards = dataset["rewards"]
        terminals = dataset["terminals"]
        realterminals = dataset["realterminals"]

        # TODO: check std differences again
        obs_mean, obs_std = mean(observations, dims=2)  , std(observations, dims=2, corrected=false)
        act_mean, act_std = mean(actions, dims=2), std(actions, dims=2, corrected=false)
        reward_mean, reward_std = mean(rewards), std(rewards, corrected=false)

        if normalize_raw
            observations = (observations .- obs_mean) ./ obs_std
            actions = (actions .- act_mean) ./ act_std
        end

        observations_raw = observations
        actions_raw = actions
        joined_raw = cat(observations, actions, dims=1) # join at the last dim
        rewards_raw = rewards
        terminals_raw = terminals

        if penalty !== nothing #TODO: this should be true, handle args
            terminal_mask = squeeze(realterminals)
            rewards_raw[terminal_mask] = penalty
        end

        println("[ datasets/sequence ] Segmenting...")
        joined_segmented, termination_flags, path_lengths = segment(joined_raw, terminals, max_path_length)
        rewards_segmented, _, _ = segment(rewards_raw, terminals, max_path_length)
        println('✓')

        discounts = reshape(discount .^ collect(0:max_path_length-1), 1, :)
        values_segmented = zeros(Float32, size(rewards_segmented)...)
        for t in 1:max_path_length
            V = sum(rewards_segmented[:, t+1:end, :] .* discounts[:, 1:end-t], dims=2)
            values_segmented[:, t] = V
        end

        new(env, sequence_length, step, max_path_length, device, disable_goal)
    end
end


env = "halfcheetah-medium-expert-v2"
dataset = SequenceDataset(env)

end
