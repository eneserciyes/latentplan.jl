module Setup

using Configurations: to_dict

export watch
function watch(args_to_watch::Vector{Tuple{String, String}})::Function
    function _fn(args)
        exp_name = []
        for (key, label) in args_to_watch
            if !hasproperty(args, key)
                continue
            end
            
            val = getfield(args, key)
            push!(exp_name, "$label$val")
        end
        exp_name = join(exp_name, "_")
        exp_name = replace(exp_name, "/_" => "/")
        
    end
    return _fn
end


export parser
function parser(args::Dict{String, Any}; experiment::Union{String, Nothing}=nothing)
    if !haskey(args, "config")
        return args
    end
    params = to_dict(read_config(args, experiment))

    args = merge(params, args)
    # TODO: set seed, generate expression name, mkdir and rest
end


function read_config(args::Dict{String, Any}, experiment::Union{String, Nothing})
    dataset = replace(args["dataset"], "-" => "_")
    config = args["config"]
    print("[ utils/setup ] Reading config: $config:$dataset")
    config_module = include(config)
    if hasproperty(config_module, Symbol(dataset))
        params = getproperty(config_module, Symbol(dataset))[experiment]
        print("Overriding base configs with $dataset configs.")
    else
        if experiment == "train"
            params = config_module.Train()
        elseif experiment == "plan"
            params = config_module.Plan()
        end
    end
    return params
end

end
