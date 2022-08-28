module D4RL

using PyCall

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
function qlearning_dataset_with_timeouts(env; dataset=nothing, terminate_on_end::Bool=false, disable_goal::Bool=false, kwargs...)
    if dataset === nothing
        dataset = env.get_dataset(kwargs)
    end
    N = dataset["rewards"].shape[0]


end
end
