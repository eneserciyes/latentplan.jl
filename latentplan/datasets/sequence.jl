module Sequence
include("d4rl.jl")
include("../vector_utils.jl")

using .D4RL
using .Vutils: squeeze
using Printf
using Statistics: mean, std


function segment(observations, terminals, max_path_length)
    @assert length(observations) == length(terminals)
    observation_dim = size(observations, 2)
    trajectories = [[]]

    for (obs, term) in zip(observations, terminals)
        push!(trajectories[end], obs)
        if squeeze(term)
            push!(trajectories, [])
        end 
    end

    if length(trajectories[end]) == 0
        trajectories = trajectories[:end-1]
    end

    n_trajectories = length(trajectories)
    path_lengths = [length(traj) for traj in trajectories]

    # pad trajectories to be equal length
    trajectories_pad = zeros(trajectories[1].dtype, (n_trajectories, max_path_length, observation_dim))
    early_termination = zeros(Bool, (n_trajectories, max_path_length))
    for (i, traj) in enumerate(trajectories)
        path_length = path_lengths[i]
        trajectories_pad[i, 1:path_length] = traj
        early_termination[i, path_length:end] = 1
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

        dataset = qlearning_dataset_with_timeouts(env.unwrapped, terminate_on_end=true, disable_goal=disable_goal)
        print('✓')

        # TODO: preprocess_fn

        ##
        observations = dataset["observations"]
        actions = dataset["actions"]
        rewards = dataset["rewards"]
        terminals = dataset["terminals"]
        realterminals = dataset["realterminals"]

        obs_mean, obs_std = mean(observations, dims=1), std(observations, dims=1)
        act_mean, act_std = mean(actions, dims=1), std(actions, dims=1)
        reward_mean, reward_std = mean(rewards, dims=1), std(rewards, dims=1)

        if normalize_raw
            observations = (observations .- obs_mean) ./ obs_std
            actions = (actions .- act_mean) ./ act_std
        end

        observations_raw = observations
        actions_raw = actions
        joined_raw = cat(observations, actions, dims=ndims(observations)) # join at the last dim
        rewards_raw = rewards
        terminals_raw = terminals

        if penalty !== nothing
            terminal_mask = squeeze(realterminals)
            rewards_raw[terminal_mask] = penalty
        end

        println("[ datasets/sequence ] Segmenting...")
        joined_segmented, termination_flags, path_lengths = segment(joined_raw, terminals, max_path_length)
        rewards_segmented, _, _ = segment(rewards_raw, terminals, max_path_length)
        println('✓')

        discounts = reshape(discount .^ collect(1:max_path_length), :, 1)
        values_segmented = zeros(Float32, size(rewards_segmented)...)
        for t in 1:max_path_length
            V = sum(rewards_segmented[:, t+1:end] .* discounts[1:end-t-1], dims=2)
            values_segmented[:, t] = V
        end

        new(env, sequence_length, step, max_path_length, device, disable_goal)
    end
end


env = "halfcheetah-medium-expert-v2"
dataset = SequenceDataset(env)

end
