using PyCall
using ProgressMeter: @showprogress

d4rl = pyimport("d4rl")
gym = pyimport("gym")

export load_environment
function load_environment(name::String)
    wrapped_env = gym.make(name)
    env = wrapped_env.unwrapped
    env.max_episode_steps = wrapped_env._max_episode_steps
    env.name = name
    env
end

export qlearning_dataset_with_timeouts
function qlearning_dataset_with_timeouts(env; dataset=nothing, terminate_on_end::Bool=false, disable_goal::Bool=false, debug::Bool=false, kwargs...)
    if dataset === nothing
        dataset = env.get_dataset(;kwargs...)
    end
    N = size(dataset["rewards"])[end]
    obs_ = Vector{Vector{Float32}}(undef, N-1)
    next_obs_ = Vector{Vector{Float32}}(undef, N-1)
    action_ = Vector{Vector{Float32}}(undef, N-1)
    reward_ = Vector{Float32}(undef, N-1)
    done_ = Vector{Bool}(undef, N-1)
    realdone_ = Vector{Bool}(undef, N-1)
    if haskey(dataset, "infos/goal")
        # TODO: later
        if !disable_goal
            print("Goal enabled")
        else
            print("Goal disabled")
        end
    end

    episode_step = 0
    for i = 1:N-1
        obs = dataset["observations"][i,:]
        new_obs = dataset["observations"][i+1,:]
        action = dataset["actions"][i,:]
        reward = dataset["rewards"][i]
        done_bool = Bool.(dataset["terminals"][i])   
        realdone_bool = Bool.(dataset["terminals"][i])
        # TODO
        if haskey(dataset, "infos/goal")
            final_timestep = any(dataset["infos/goal"][i] .!= dataset["infos/goal"][i+1])
        else
            final_timestep=dataset["timeouts"][i]
        end

        if i < N
            done_bool += final_timestep
        end

        if !terminate_on_end && final_timestep
            episode_step = 0
            continue
        end
        if done_bool>0 || final_timestep
            episode_step=0
        end

        obs_[i] = obs
        next_obs_[i] = new_obs
        action_[i] = action
        reward_[i] = reward
        done_[i] =  done_bool
        realdone_[i] = realdone_bool
        episode_step += 1
    end

    return Dict(
        "observations"=>reduce(hcat, obs_),
        "actions"=>reduce(hcat, action_),
        "next_observations"=>reduce(hcat, next_obs_),
        "rewards"=> reshape(reward_, 1, :),
        "terminals"=>done_,
        "realterminals"=>realdone_,
    )
end

