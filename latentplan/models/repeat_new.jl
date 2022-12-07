function repeat_broadcast(A, r...)
    if length(r) > ndims(A)
        A = reshape(A, size(A)..., ones(Int, length(r) - ndims(A))...)
    end
    if length(r) == 0 
        return A
    end
    if r[1] == 1
        return repeat_broadcast(A, r[2:end]...)
    else
        return repeat_broadcast(_repeat_broadcast(A, ndims(A) - length(r) + 1, r[1]), r[2:end]...)
    end
end

function repeat_interleave(A, r...)
    if length(r) == 0 
        return A
    end
    if r[1] == 1
        return repeat_interleave(A, r[2:end]...)
    else
        return repeat_interleave(_repeat_interleave(A, ndims(A) - length(r) + 1, r[1]), r[2:end]...)
    end
end


function _repeat_broadcast(A,x,r)
    Asize = size(A)
    Ahat = reshape(A, Asize[1:x]..., 1, Asize[x+1:end]...) .* ones(eltype(A), Asize[1:x]..., r, Asize[x+1:end]...)
    return reshape(Ahat, Asize[1:x-1]..., Asize[x]*r, Asize[x+1:end]...)
end

function _repeat_interleave(A, x, r)
    Asize = size(A)
    Ahat = reshape(A, Asize[1:x-1]..., 1, Asize[x:end]...) .* ones(eltype(A), Asize[1:x-1]..., r, Asize[x:end]...)
    return reshape(Ahat, Asize[1:x-1]..., Asize[x]*r, Asize[x+1:end]...)
end
