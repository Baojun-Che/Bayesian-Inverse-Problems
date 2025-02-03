using Random
using PyPlot
using Distributions
using LinearAlgebra
using Statistics
using DocStringExtensions
include("LowRankCov.jl")
include("QuadratureRule.jl")
include("GaussianMixture.jl")

"""
Particles-based Gaussian Mixture Black-Box Variational Inference.
"""
mutable struct PBBVIObj{FT<:AbstractFloat, IT<:Int}
    "object name"
    name::String
    "a vector of arrays of size (N_modes) containing the modal weights of the parameters"
    logx_w::Vector{Array{FT, 1}} 
    "a vector of arrays of size (N_modes x N_x) containing the modal means of the parameters"
    x_mean::Vector{Array{FT, 2}} 
    "a vector of arrays of size (N_modes x N_x x N_r) containing the (pseudo) sqrt covariances of the parameters"
    xx_sqrt_cov::Union{Vector{Array{FT, 3}}, Nothing}
    "number of modes"
    N_modes::IT
    "size of x"
    N_x::IT
    "number of sampling points (to compute expectation using MC)"
    N_r::IT
    "current iteration number"
    iter::IT
    "single_Gaussian, Gaussian_mixture"
    random_quadrature_type::String
    "weight clipping"
    w_min::FT
    "eps for C=QQ'+eps*I"
    cov_eps::FT
end


function PBBVIObj(
                x0_w::Array{FT, 1},
                x0_mean::Array{FT, 2},
                xx0_sqrt_cov::Union{Array{FT, 3}, Nothing};
                random_quadrature_type::String = "Gaussian_mixture",
                w_min::FT = 1.0e-8,
                cov_eps::FT = 0.1) where {FT<:AbstractFloat}

    N_modes, N_x, N_r = size(xx0_sqrt_cov)

    logx_w = Array{FT,1}[]      # array of Array{FT, 1}'s
    push!(logx_w, log.(x0_w))   # insert parameters at end of array (in this case just 1st entry)
    x_mean = Array{FT,2}[]      # array of Array{FT, 2}'s
    push!(x_mean, x0_mean)      # insert parameters at end of array (in this case just 1st entry)
    xx_sqrt_cov = Array{FT,3}[]      # array of Array{FT, 3}'s
    push!(xx_sqrt_cov, xx0_sqrt_cov)      # insert parameters at end of array (in this case just 1st entry)
    
    name = "P-GM-BBVI"

    iter = 0

    PBBVIObj(name,
            logx_w, x_mean, xx_sqrt_cov, N_modes, N_x, N_r,
            iter, random_quadrature_type,  w_min, cov_eps)
end


function sqrt_cov_update(Q, R, D; inv_eps::Real=0.01) 
    """
        For Q=sqrt_xx_cov, R=x_p, D=diag(log_ratio) , efficiently calculate: 
        Q - (1/2)*RDR'Q(Q'Q+inv_eps)^{-1}.
    """
    # abs(sum(D))<0.1
    N_ens = size(D,1)
    D = D./N_ens
    temp1 = Q'*Q + inv_eps*I
    temp2 = temp1\(Q')
    temp3 = (temp2*R)'
    return Q - R*diagm(D)*temp3/2
end
   
""" func_Phi: the potential function, i.e the posterior is proportional to exp( - func_Phi)"""
function update_ensemble!(gmgd::PBBVIObj{FT, IT}, func_Phi::Function, dt_max::FT) where {FT<:AbstractFloat, IT<:Int}
    
    # @show gmgd.iter
    gmgd.iter += 1
    N_x,  N_modes = gmgd.N_x, gmgd.N_modes
    N_r = gmgd.N_r
    eps = gmgd.cov_eps

    x_mean  = gmgd.x_mean[end]
    logx_w  = gmgd.logx_w[end]
    xx_sqrt_cov  = gmgd.xx_sqrt_cov[end]
    x_w = exp.(logx_w)
    x_w ./= sum(x_w)

    random_quadrature_type=gmgd.random_quadrature_type

    d_logx_w, d_x_mean = zeros(N_modes), zeros(N_modes, N_x)
    R_list = []
    D_list = []

    if random_quadrature_type == "single_Gaussian"

        N_ens = 2*N_r
        for im = 1:N_modes 

            log_ratio = zeros(N_ens)
            R = xx_sqrt_cov[im,:,:]*sqrt(N_r)+sqrt(eps)*randn(N_x,N_r)*0
            R = hcat(R,-R) # R=[x_1-m,x_2-m,...]

            for i = 1:N_ens
                for imm = 1:N_modes
                    log_ratio[i] += x_w[imm]*degenerate_Gaussian_density(R[:,i]+x_mean[im,:]-x_mean[imm,:], xx_sqrt_cov[imm,:,:]; eps=eps) 
                end
                log_ratio[i] = log(log_ratio[i])+func_Phi(R[:,i]+x_mean[im,:])
            end

            # E[logρ+Phi]
            log_ratio_mean = mean(log_ratio)

            log_ratio.-=log_ratio_mean
            # -E[(x-m)(logρ+Phi)]
            d_x_mean[im,:] = -mean( R[:,i]*log_ratio[i] for i=1:N_ens)   
            d_logx_w[im] = -log_ratio_mean

            push!(R_list, R)
            push!(D_list, log_ratio)
        end
    else 
        @error "UNDEFINED random_quadrature_type!"
    end

    dt = dt_max
    for im = 1:N_modes
        dt = min(dt, size(D_list[1],1)*0.5/maximum(abs.(D_list[im])))
    end
    if gmgd.iter%10==0   @show dt  end 

    x_mean_n = copy(x_mean) 
    xx_sqrt_cov_n = copy(xx_sqrt_cov)
    logx_w_n = copy(logx_w)

    for im = 1:N_modes
        x_mean_n[im,:] += dt*d_x_mean[im,:]
        logx_w_n[im] += dt*d_logx_w[im]
        xx_sqrt_cov_n[im,:,:] = sqrt_cov_update(xx_sqrt_cov_n[im,:,:],R_list[im],dt*D_list[im])
    end

    # Normalization
    w_min = gmgd.w_min
    logx_w_n .-= maximum(logx_w_n)
    logx_w_n .-= log( sum(exp.(logx_w_n)) )
    x_w_n = exp.(logx_w_n)
    clip_ind = x_w_n .< w_min
    x_w_n[clip_ind] .= w_min
    x_w_n[(!).(clip_ind)] /= (1 - sum(clip_ind)*w_min)/sum(x_w_n[(!).(clip_ind)])
    logx_w_n .= log.(x_w_n)
    
    
    ### Save results
    push!(gmgd.x_mean, x_mean_n)
    push!(gmgd.xx_sqrt_cov, xx_sqrt_cov_n)
    push!(gmgd.logx_w, logx_w_n) 
end


##########
function Gaussian_mixture_PBBVI(func_Phi, x0_w, x0_mean, xx0_sqrt_cov;
    random_quadrature_type::String = "single_Gaussian", N_iter = 100, dt = 5.0e-1, N_r = -1)

    N_modes , N_x = size(x0_mean)
    if random_quadrature_type == "Gaussian_mixture"
        if N_r == -1  N_r = N_modes * N_x  end
    elseif random_quadrature_type == "single_Gaussian"
        if N_r == -1  N_r = ceil(sqrt(N_x))  end
    else 
        @error "UNDEFINED random_quadrature_type in BBVI"
    end

    gmgdobj=PBBVIObj(
        x0_w, x0_mean, xx0_sqrt_cov;
        random_quadrature_type = random_quadrature_type)

    for i in 1:N_iter
        if i%max(1, div(N_iter, 10)) == 0  @info "iter = ", i, " / ", N_iter  end
        
        update_ensemble!(gmgdobj, func_Phi, dt) 
    end
    
    return gmgdobj
end
