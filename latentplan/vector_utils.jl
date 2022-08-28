module Vutils

export squeeze
function squeeze(A::AbstractArray)
    singleton_dims = tuple((d for d in 1:ndims(A) if size(A, d) == 1)...)
    return dropdims(A, dims=singleton_dims)
end

end
