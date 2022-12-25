function dropdims_n(A;dims::Tuple)
    sz = [s for s in size(A)]
    @assert all([size(A, d) for d in dims] .== 1) "All dimensions to drop must be of size 1"
    deleteat!(sz, dims)
    reshape(A, sz...)
end
