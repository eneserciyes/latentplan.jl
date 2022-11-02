module Embeddings
export SmoothEmbedding

function make_weights(N, weights)
end

function add_stop_token(tokens)
end

struct SmoothEmbedding
    weights;
    inds;
    _embeddings;

    function SmoothEmbedding(num_embeddings, embedding_dim, weights, stop_token=false)
    end

end

function (s::SmoothEmbedding)(x)

end

end
