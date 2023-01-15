export DataLoader, get_test, get_item, SequenceDataset, denormalize, normalize_joined_single, denormalize_joined

include("d4rl.jl")

using Printf
using Statistics: mean, std
using ProgressMeter: @showprogress
using Random: shuffle
using Debugger: @bp

function squeeze(A::AbstractArray)
    singleton_dims = tuple((d for d in 1:ndims(A) if size(A, d) == 1)...)
    return dropdims_n(A, dims=singleton_dims)
end

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
        trajectories = trajectories[1:end-1]
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

function compute_values(rewards_segmented, discounts, max_path_length)
    values_segmented = zeros(Float32, size(rewards_segmented)...)
    @showprogress "Calculating values" for t in 1:max_path_length
        V = sum(rewards_segmented[:, t+1:end, :] .* discounts[:, 1:end-t], dims=2)
        values_segmented[:, t, :] = V
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
    atype;

    function SequenceDataset(env; sequence_length=250, step=10, 
        discount=0.99, max_path_length=1000,
        penalty=nothing, device::String="cuda:0", normalize_raw::Bool=true, normalize_reward::Bool=true,
        train_portion=1.0, disable_goal::Bool=false, atype=Knet.atype())
    
        @printf("[ datasets/sequence ] Sequence length: %d | Step: %d | Max path length: %d\n", sequence_length, step, max_path_length)
        
        env = typeof(env) == String ? load_environment(env) : env
        println("[ datasets/sequence ] Loading...")

        dataset = qlearning_dataset_with_timeouts(env.unwrapped, terminate_on_end=true, disable_goal=disable_goal, debug=false)
        println('✓')

        ##
        observations = dataset["observations"]
        actions = dataset["actions"]
        rewards = dataset["rewards"]
        terminals = dataset["terminals"]
        realterminals = dataset["realterminals"]

        @bp
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

        if penalty !== nothing
            terminal_mask = squeeze(realterminals)
            rewards_raw[terminal_mask] .= penalty
        end

        println("[ datasets/sequence ] Segmenting...")
        joined_segmented, termination_flags, path_lengths = segment(joined_raw, terminals, max_path_length)
        rewards_segmented, _, _ = segment(rewards_raw, terminals, max_path_length)
        println('✓')

        discounts = reshape(discount .^ collect(0:max_path_length-1), 1, :)
        values_segmented = compute_values(rewards_segmented, discounts, max_path_length)
        
        values_raw = reshape(dropdims_n(values_segmented, dims=tuple(1)), :)
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
            e = l - 1
            split = trunc(Int, e * train_portion)
            for i in 1:e
                if i <= split
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
            observation_dim, action_dim, joined_dim, atype
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
    joined_std = s.atype(vcat(s.obs_std[:, 1], s.act_std[:, 1], [s.reward_std;], [s.value_std;]))
    joined_mean = s.atype(vcat(s.obs_mean[:, 1], s.act_mean[:, 1], [s.reward_mean;], [s.value_mean;]))
    return (joined .- joined_mean) ./ joined_std
end

function denormalize_joined(s::SequenceDataset, joined)
    states = joined[1:s.observation_dim, :]
    actions = joined[s.observation_dim:s.observation_dim+s.action_dim, :]
    rewards = reshape(joined[end-2, :], 1, size(joined, 2), 1)
    values = reshape(joined[end-1, :], 1, size(joined, 2), 1)
    results = denormalize(s, states, actions, rewards, values)
    return cat(tuple(results..., reshape(joined[end, :], 1, size(joined, 2), 1)), dims=1)
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
    path_ind, start_ind, end_ind = s.indices[idx]
    joined = s.joined_segmented[:, start_ind:s.step:end_ind-1, path_ind]
    
    traj_inds = Vector(start_ind:s.step:end_ind-1)
    mask = ones(Bool, size(joined))
    mask[:, traj_inds .>= s.max_path_length - s.step + 1] .= 0
    terminal = (.~cumprod(.~(reshape(s.termination_flags[start_ind:s.step:end_ind-1, path_ind], 1, :, 1)), dims=1))[:,:,1]
    X = convert(s.atype, joined[:, 1:end-1])
    Y = convert(s.atype, joined[:, 2:end])
    mask = mask[:,1:end-1]
    terminal = terminal[:, 1:end-1]
    return X, Y, mask, terminal
end

function get_test(s::SequenceDataset)
    Xs = []
    Ys = []
    masks = []
    terminals = []
    for (path_ind, start_ind, end_ind) in s.test_indices
        joined = s.joined_segmented[:, start_ind:s.step:end_ind-1, path_ind]
        traj_inds = Vector(start_ind:s.step:end_ind-1)
        mask = ones(Bool, size(joined))
        mask[:, traj_inds .>= s.max_path_length - s.step + 1] .= 0 
        terminal = (.~cumprod(.~(reshape(s.termination_flags[start_ind:s.step:end_ind-1, path_ind], 1, :, 1)), dims=1))[:,:,1]
        X = joined[:, 1:end-1]
        Y = joined[:, 2:end]
        mask = mask[:,1:end-1]
        terminal = terminal[:, 1:end-1]
        push!(Xs, X)
        push!(Ys, Y)
        push!(masks, mask)
        push!(terminals, terminal)
    end

    return cat(Xs..., dims=3), cat(Ys..., dims=3), cat(masks..., dims=3), cat(terminals..., dims=3)
end

struct DataLoader
    dataset::SequenceDataset
    batch_size::Int
    shuffle::Bool

    function DataLoader(dataset::SequenceDataset; batch_size::Int, shuffle::Bool)
        new(dataset, batch_size, shuffle)
    end
end


function Base.length(d::DataLoader)
    return length(d.dataset) ÷ d.batch_size
end

function Base.iterate(d::DataLoader)
    indices = 1:length(d.dataset)
    if d.shuffle
        indices = shuffle(indices)
    end
    idx = 1
    state = (idx, indices)
    return iterate(d, state)
end

function Base.iterate(d::DataLoader, state)
    idx, indices = state
    if idx > length(d.dataset)
        return nothing
    end
    Xs = []
    Ys = []
    masks = []
    terminals = []
    for i in idx:min(idx+d.batch_size-1, length(d.dataset))
        X, Y, mask, terminal = get_item(d.dataset, indices[i])
        push!(Xs, X)
        push!(Ys, Y)
        push!(masks, mask)
        push!(terminals, terminal)
    end
    X = cat(Xs..., dims=3)
    Y = cat(Ys..., dims=3)
    mask = cat(masks..., dims=3)
    terminal = cat(terminals..., dims=3)
    state = (idx+d.batch_size, indices)
    return (X, Y, mask, terminal), state
end
