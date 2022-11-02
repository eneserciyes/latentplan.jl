module Ein

    export EinLinear
    
    struct EinLinear
        n_models;
        out_features;
        in_features;
        weight;
        bias;
        function EinLinear(n_models,  in_features, out_features, bias)
            # TODO: first init, then reset_parameters function here
        end
    end

    function (e::EinLinear)(input)
    end

    function (e::EinLinear)()
        # TODO: extra_repr function here
    end
end


