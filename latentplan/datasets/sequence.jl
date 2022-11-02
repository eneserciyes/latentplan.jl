module Sequence
include("d4rl.jl")
include("../vector_utils.jl")

using .D4RL
using .Vutils: squeeze
using Printf
using Statistics: mean, std
using ProgressMeter: @showprogress


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
        early_termination[path_length+1:end, i] .= 1
    end
    return trajectories_pad, early_termination, path_lengths
end 

function compute_values(rewards_segmented::Array{Float32, 3}, discounts::Matrix{Float32}, max_path_length::Int32)
    values_segmented = zeros(Float32, size(rewards_segmented)...)
    @showprogress "Calculating values" for t in 1:max_path_length
        V = sum(rewards_segmented[:, t+1:end, :] .* discounts[:, 1:end-t], dims=2)
        values_segmented[:, t] = V
    end
    return values_segmented
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
    observations_raw;
    actions_raw;
    joined_raw;
    rewards_raw;
    terminals_raw;
    joined_segmented; 
    termination_flags;
    path_lengths; 
    rewards_segmented;
    discount;
    discounts;
    values_segmented;
    values_raw;
    value_mean; value_std;
    train_portion;
    test_portion;
    indices;
    test_indices;
    observation_dim;
    action_dim;
    joined_dim;

    function SequenceDataset(env; sequence_length::Int32=Int32(250), step::Int32=Int32(10), 
        discount::Float32=0.99f0, max_path_length::Int32=Int32(1000),
        penalty::Int32=nothing, device::String="cuda:0", normalize_raw::Bool=true, normalize_reward::Bool=true,
        train_portion::Float32=1.0f0, disable_goal::Bool=false)
    
        @printf("[ datasets/sequence ] Sequence length: %d | Step: %d | Max path length: %d\n", sequence_length, step, max_path_length)
        
        env = typeof(env) == String ? load_environment(env) : env
        println("[ datasets/sequence ] Loading...")

        dataset = qlearning_dataset_with_timeouts(env.unwrapped, terminate_on_end=true, disable_goal=disable_goal, debug=true)
        println('✓')

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
            rewards_raw[terminal_mask] .= penalty
        end

        println("[ datasets/sequence ] Segmenting...")
        joined_segmented, termination_flags, path_lengths = segment(joined_raw, terminals, max_path_length)
        rewards_segmented, _, _ = segment(rewards_raw, terminals, max_path_length)
        println('✓')

        discounts = reshape(discount .^ collect(0:max_path_length-1), 1, :)
        values_segmented = compute_values(rewards_segmented, discounts, max_path_length)

        values_raw = reshape(dropdims(values_segmented, dims=ndims(values_segmented)), :)
        values_mask = .!reshape(termination_flags, :)
        values_raw = reshape(values_raw[values_mask], 1, :)

        if normalize_raw && normalize_reward
            value_mean, value_std = mean(values_raw), std(values_raw, corrected=false)
            values_raw = (values_raw .- value_mean) ./ value_std
            rewards_raw = (rewards_raw .- reward_mean) ./ reward_std
            
            values_segmented = (values_segmented .- value_mean) ./ value_std
            rewards_segmented = (rewards_segmented .- reward_mean) ./ reward_std
        else
            value_mean, value_std = 0.0, 1.0
        end

        joined_raw = cat(joined_raw, rewards_raw, values_raw; dims=1)
        joined_segmented = cat(joined_segmented, rewards_segmented, values_segmented, dims=1)

        test_portion = 1.0 - train_portion

        ## get valid indices
        indices = []
        test_indices = []
        for (path_ind, l) in enumerate(path_lengths)
            e = l
            split = trunc(Int, e * train_portion)
            for i in 1:split
                if i < split
                    push!(indices, (path_ind, i, i+sequence_length))
                else
                    push!(test_indices, (path_ind, i, i+sequence_length))
                end
            end
        end

        observation_dim = size(observations, ndims(observations)-1)
        action_dim = size(actions, ndims(actions)-1)
        joined_dim = size(joined_raw, ndims(joined_raw)-1)

        ## pad trajectories
        dim_joined, _, n_trajectories = size(joined_segmented)
        joined_segmented = cat(joined_segmented, zeros(Float32, dim_joined, sequence_length-1, n_trajectories), dims=2)
        termination_flags = cat(termination_flags, ones(Bool, sequence_length-1, n_trajectories), dims=1)


        new(
            env, sequence_length, step, max_path_length, 
            device, disable_goal, normalize_raw, normalize_reward, 
            obs_mean, obs_std, act_mean, act_std, reward_mean, reward_std,
            observations_raw, actions_raw, joined_raw, rewards_raw, terminals_raw,
            joined_segmented, termination_flags, path_lengths, 
            rewards_segmented, discount, discounts,
            values_segmented, values_raw, value_mean, value_std,
            train_portion, test_portion, indices, test_indices,
            observation_dim, action_dim, joined_dim
        )
    end
end

function denormalize(s::SequenceDataset, states, actions, rewards, values)
    states = states .* s.obs_std + s.obs_mean
    actions = actions .* s.act_std + s.act_mean
    rewards = rewards .* s.reward_std + s.reward_mean
    values = values .* s.value_std + s.value_mean

    return states, actions, rewards, values
end

function normalize_joined_single(s::SequenceDataset, joined)
    joined_std = cat([s.obs_std[:, 1], s.act_std[:, 1], [s.reward_std], [s.value_std]])
    joined_mean = cat([s.obs_mean[:, 1], s.act_mean[:, 1], [s.reward_mean], [s.value_mean]])
    return (joined .- joined_mean) / joined_std
end

function denormalize_joined(s::SequenceDataset, joined)
end

function normalize_states(s::SequenceDataset, states)
end

function denormalize_states(s::SequenceDataset, states)
end

function denormalize_values(s::SequenceDataset, values)
end

function Base.length(s::SequenceDataset)
    return length(s.indices)
end

function get_item(s::SequenceDataset, idx)
end

function get_test(s::SequenceDataset)
end



end
