module Setup

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

end
