using Random
using PyPlot
using Distributions
using LinearAlgebra
using Statistics
using DocStringExtensions
include("LowRankCov.jl")
include("../Inversion/QuadratureRule.jl")
include("../Inversion/GaussianMixture.jl")

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
    "a vector of arrays of size (N_modes) containing the modal weights of the parameters"
    cov_eps::Vector{Array{FT, 1}} 
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
    "cov_eps lower bound"
    cov_eps_min::FT
end


function PBBVIObj(
                x0_w::Array{FT, 1},
                x0_mean::Array{FT, 2},
                xx0_sqrt_cov::Union{Array{FT, 3}, Nothing},
                cov_eps0::Array{FT, 1};
                random_quadrature_type::String = "Gaussian_mixture",
                w_min::FT = 1.0e-8,
                cov_eps_min::FT = 0.01) where {FT<:AbstractFloat}

    N_modes, N_x, N_r = size(xx0_sqrt_cov)

    logx_w = Array{FT,1}[]      # array of Array{FT, 1}'s
    push!(logx_w, log.(x0_w))   # insert parameters at end of array (in this case just 1st entry)
    x_mean = Array{FT,2}[]      # array of Array{FT, 2}'s
    push!(x_mean, x0_mean)      # insert parameters at end of array (in this case just 1st entry)
    xx_sqrt_cov = Array{FT,3}[]      # array of Array{FT, 3}'s
    push!(xx_sqrt_cov, xx0_sqrt_cov)      # insert parameters at end of array (in this case just 1st entry)
    cov_eps = Array{FT,1}[]    # array of Array{FT, 1}'s
    push!(cov_eps, cov_eps0)   # insert parameters at end of array (in this case just 1st entry)

    name = "EnBBVI"

    iter = 0

    PBBVIObj(name,
            logx_w, x_mean, xx_sqrt_cov, cov_eps,
            N_modes, N_x, N_r,
            iter, random_quadrature_type,  w_min, cov_eps_min)
end


function low_rank_approximation(eps0, Q0, R0, D0, N_r; rank_plus::Int = 1, A = nothing, eps_min = 0.01)
    """Approximate A = eps0*I + Q0*Q0' + R0*D0*R0'  with  A = eps*I +QQ',
    based on PPCA method and random SVD. """
    if A == nothing
        N_x = size(Q0,1)
        Omega = randn(N_x, N_r+rank_plus)
        Y = eps0*Omega + Q0*(Q0'*Omega) + (R0.*D0')*(R0'*Omega)
        Yqr = qr(Y)
        Q = Matrix(Yqr.Q)  
        B = eps0*Q' + Q'*Q0*Q0' + Q'*(R0.*D0')*R0'
        trA = eps0*N_x + sum(Q0.*Q0) + sum(R0.*(R0.*D0'))
    else
        trA = tr(A)
        N_x = size(A,1)
        Omega = randn(N_x, N_r+rank_plus)
        Y = A*Omega
        Yqr = qr(Y)
        Q = Matrix(Yqr.Q)  
        B = Q'*A
    end 
    U0, D, _ = svd(B)
    U = (Q*U0)[:,1:N_r]
    D = D[1:N_r]
    eps = (trA-sum(D))/(N_x-N_r)
    eps = max(eps, eps_min)
    newQ = hcat( [sqrt(max(D[i]-eps,0.01))*U[:,i] for i=1:N_r]... )
    return eps, newQ
end
   
""" func_Phi: the potential function, i.e the posterior is proportional to exp( - func_Phi)"""
function update_ensemble!(gmgd::PBBVIObj{FT, IT}, func_Phi::Function, dt_max::FT) where {FT<:AbstractFloat, IT<:Int}
    

    gmgd.iter += 1
    N_x,  N_modes = gmgd.N_x, gmgd.N_modes
    N_r = gmgd.N_r
    cov_eps_min = gmgd.cov_eps_min 

    x_mean  = gmgd.x_mean[end]
    logx_w  = gmgd.logx_w[end]
    xx_sqrt_cov  = gmgd.xx_sqrt_cov[end]
    cov_eps = gmgd.cov_eps[end]

    # if gmgd.iter%10==0
    #     @show  gmgd.iter
    #     @show  maximum(cov_eps)
    #     @show  maximum([norm(x_mean[im,:])  for im = 1:N_modes])
    #     @show  maximum([norm(xx_sqrt_cov[im,:,:])  for im = 1:N_modes])
    # end
    x_w = exp.(logx_w)
    x_w ./= sum(x_w)

    random_quadrature_type=gmgd.random_quadrature_type

    cov_pseudo_inv = []
    for im = 1:N_modes
        cov_pseudo_im = xx_sqrt_cov[im,:,:]'*xx_sqrt_cov[im,:,:]+cov_eps[im]*I
        push!(cov_pseudo_inv, inv(cov_pseudo_im)) 
    end

    d_logx_w, d_x_mean = zeros(N_modes), zeros(N_modes, N_x)
    R_list = []
    D_list = []

    if random_quadrature_type == "single_Gaussian"

        N_ens = 2*N_r
        for im = 1:N_modes 

            log_ratio = zeros(N_ens)
            R = xx_sqrt_cov[im,:,:]*randn(N_r,N_r)+sqrt(cov_eps[im])*randn(N_x,N_r)
            R = hcat(R,-R) # R=[x_1-m,x_2-m,...]
            
            for i = 1:N_ens
                for imm = 1:N_modes
                    log_ratio[i] += x_w[imm]*Low_Rank_Gaussian_density(R[:,i]+x_mean[im,:]-x_mean[imm,:], xx_sqrt_cov[imm,:,:], cov_eps[imm]; D_inv=cov_pseudo_inv[imm]) 
                end
                log_ratio[i] = log(log_ratio[i])+func_Phi(R[:,i]+x_mean[im,:])
            end

            # E[logρ+Phi]
            log_ratio_mean = mean(log_ratio)

            log_ratio.-=log_ratio_mean
            # -E[(x-m)(logρ+Phi)]
            d_x_mean[im,:] = -mean( R[:,i]*log_ratio[i] for i=1:N_ens)   
            d_logx_w[im] = -log_ratio_mean

            D = - [log_ratio[i]+log_ratio[i+N_r] for i=1:N_r]/N_ens


            push!(R_list, R[:,1:N_r])
            push!(D_list, D)
        end
    else 
        @error "UNDEFINED random_quadrature_type!"
    end

    dt = dt_max
    # Adaptive Time Step 
    for im = 1:N_modes
        R = R_list[im]
        D = D_list[im]
        N_ens = size(D,1)
        D_pos_sqrt = [sqrt(max(D[i],0))  for i = 1:N_ens]
        temp1 = R.*(D_pos_sqrt')
        temp2 = xx_sqrt_cov[im,:,:]'*temp1
        C = temp1'*temp1-temp2'*cov_pseudo_inv[im]*temp2
        dt = min(dt, 0.9*cov_eps[im]/opnorm(C,2))
    end
    if gmgd.iter%20==0   @show dt  end

    x_mean_n = copy(x_mean) 
    xx_sqrt_cov_n = copy(xx_sqrt_cov)
    logx_w_n = copy(logx_w)
    cov_eps_n = copy(cov_eps)

    for im = 1:N_modes
        x_mean_n[im,:] += dt*d_x_mean[im,:]
        logx_w_n[im] += dt*d_logx_w[im]
        cov_eps_n[im], xx_sqrt_cov_n[im,:,:] = low_rank_approximation(cov_eps[im], xx_sqrt_cov[im,:,:], R_list[im], dt*D_list[im], N_r; eps_min = cov_eps_min)
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
    
    for im = 1:N_modes 
        cov_eps_n[im] = max(cov_eps_n[im], cov_eps_min)
    end
    ### Save results
    push!(gmgd.x_mean, x_mean_n)
    push!(gmgd.xx_sqrt_cov, xx_sqrt_cov_n)
    push!(gmgd.logx_w, logx_w_n)
    push!(gmgd.cov_eps, cov_eps_n) 

end


##########
function Gaussian_mixture_EnBBVI(func_Phi, x0_w, x0_mean, xx0_sqrt_cov, cov_eps0;
    random_quadrature_type::String = "single_Gaussian", N_iter = 100, dt = 5.0e-1)


    gmgdobj=PBBVIObj(
        x0_w, x0_mean, xx0_sqrt_cov, cov_eps0;
        random_quadrature_type = random_quadrature_type)

    for i in 1:N_iter
        if i%max(1, div(N_iter, 10)) == 0  @info "iter = ", i, " / ", N_iter  end
        
        update_ensemble!(gmgdobj, func_Phi, dt) 
    end
    
    return gmgdobj
end
