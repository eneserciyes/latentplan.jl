using LinearAlgebra: norm, lmul!, axpy!
import Knet.Train20: update!, minimize, gclip!, full, gclip_update!, _update!

mutable struct AdamW
    lr::AbstractFloat
    beta1::AbstractFloat
    beta2::AbstractFloat
    eps::AbstractFloat
    t::Int
    gclip::AbstractFloat
    weight_decay::AbstractFloat
    fstm
    scndm
end

AdamW(; lr=0.001, gclip=0, beta1=0.9, beta2=0.999, weight_decay=0, eps=1e-8)=AdamW(lr, beta1, beta2, eps, 0, gclip, weight_decay, nothing, nothing)
adamw(f,d;lr=0.001, gclip=0, beta1=0.9, beta2=0.999, weight_decay=0, eps=1e-8,o...)=minimize(f,d,AdamW(lr,beta1,beta2,eps,0,gclip,weight_decay,nothing,nothing);o...)
adamw!(x...;o...)=for y in adamw(x...;o...); end

clone(a::AdamW)=AdamW(a.lr,a.beta1,a.beta2,a.eps,0,a.gclip,a.weight_decay,nothing,nothing)

function _update!(w, g, p::AdamW)
    T = eltype(w)
    if p.fstm===nothing; p.fstm=zero(w); p.scndm=zero(w); end
    p.t += 1
    lmul!(p.beta1, p.fstm)
    axpy!(1-p.beta1, g, p.fstm)
    lmul!(p.beta2, p.scndm)
    axpy!(1-p.beta2, g .* g, p.scndm)
    fstm_corrected = p.fstm / T(1 - p.beta1 ^ p.t)
    scndm_corrected = p.scndm / T(1 - p.beta2 ^ p.t)
    axpy!(-p.lr, (fstm_corrected ./ (sqrt.(scndm_corrected) .+ T(p.eps))) + T(p.weight_decay) * w, w)
end

function gclip_update!(w, g, p::AdamW)
    gclip!(g, p.gclip)          # gclip! supports AutoGrad.Sparse
    g = full(g)
    _update!(w, g, p)
end

AdamW(; lr=0.001, gclip=0, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0)=AdamW(lr, beta1, beta2, eps, 0, gclip, weight_decay, nothing, nothing)
adamw(f,d;lr=0.001,gclip=0,beta1=0.9,beta2=0.999,eps=1e-8, weight_decay=0.0,o...)=minimize(f,d,AdamW(lr,beta1,beta2,eps,0,gclip,weight_decay, nothing,nothing);o...)
adamw!(x...;o...)=for y in adamw(x...;o...); end

clone(a::AdamW)=AdamW(a.lr,a.beta1,a.beta2,a.eps,0,a.gclip,a.weight_decay,nothing,nothing)

update!(w::AbstractArray{<:Number}, g::AbstractArray, p::AdamW) = gclip_update!(w, g, p)
update!(w::KnetArray, g::KnetArray, p::AdamW) = gclip_update!(w, g, p)
