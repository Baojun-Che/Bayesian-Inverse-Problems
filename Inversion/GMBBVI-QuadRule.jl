using Random
using PyPlot
using Distributions
using LinearAlgebra
using Statistics
using DocStringExtensions
include("QuadratureRule.jl")
include("GaussianMixture.jl")

mutable struct BBVIObj{FT<:AbstractFloat, IT<:Int}
    "object name"
    name::String
    "a vector of arrays of size (N_modes) containing the modal weights of the parameters"
    logx_w::Vector{Array{FT, 1}} 
    "a vector of arrays of size (N_modes x N_parameters) containing the modal means of the parameters"
    x_mean::Vector{Array{FT, 2}} 
    "a vector of arrays of size (N_modes x N_parameters x N_parameters) containing the modal covariances of the parameters"
    xx_cov::Union{Vector{Array{FT, 3}}, Nothing}
    "number of modes"
    N_modes::IT
    "size of x"
    N_x::IT
    "current iteration number"
    iter::IT
    "update covariance or not"
    update_covariance::Bool
    "weather to keep covariance matrix diagonal"
    discretize_inv_covariance::Bool
    "true: discretize from inv(C); false: discretize from C"
    diagonal_covariance::Bool
    "Cholesky, SVD"
    sqrt_matrix_type::String
    "number of sampling points (to compute expectation using MC)"
    N_ens::IT
    "weight clipping"
    w_min::FT
end


function BBVIObj(
                x0_w::Array{FT, 1},
                x0_mean::Array{FT, 2},
                xx0_cov::Union{Array{FT, 3}, Nothing};
                update_covariance::Bool = true,
                discretize_inv_covariance::Bool = true,
                diagonal_covariance::Bool = false,
                sqrt_matrix_type::String = "Cholesky",
                # setup for Gaussian mixture part
                N_ens::IT = 10,
                w_min::FT = 1.0e-8) where {FT<:AbstractFloat, IT<:Int}

    N_modes, N_x = size(x0_mean)

    logx_w = Array{FT,1}[]      # array of Array{FT, 1}'s
    push!(logx_w, log.(x0_w))   # insert parameters at end of array (in this case just 1st entry)
    x_mean = Array{FT,2}[]      # array of Array{FT, 2}'s
    push!(x_mean, x0_mean)      # insert parameters at end of array (in this case just 1st entry)
    xx_cov = Array{FT,3}[]      # array of Array{FT, 2}'s
    push!(xx_cov, xx0_cov)      # insert parameters at end of array (in this case just 1st entry)
    
    name = "BBVI"

    iter = 0

    BBVIObj(name,
            logx_w, x_mean, xx_cov, N_modes, N_x,
            iter, update_covariance, discretize_inv_covariance, diagonal_covariance, 
            sqrt_matrix_type, N_ens, w_min)
end

function random_orthogonal_matrix(N_x)
    A = randn(N_x, N_x) 
    Q = qr(A).Q      
    return Matrix(Q) 
end

function LS_approx(x_p, log_ratio, x_mean, inv_sqrt_cov, alpha)
    N_ens, N_x = size(x_p)
    # y_p = (x_p - ones(N_ens)*x_mean')*inv_sqrt_cov'
    # @show y_p
    # M = zeros(N_ens,2*N_x+1)
    # for i = 1:N_ens
    #     M[i,1:N_x] = y_p[i,:].*y_p[i,:]/2
    #     M[i,N_x+1:2*N_x] = y_p[i,:]
    #     M[i,2*N_x+1] = 1
    # end
    # # A = M'*M + 0.001*i
    # # b = M'*log_ratio
    # x = M \ (log_ratio.-mean(log_ratio)) #  A\b returns the least-square solution of Ax=b
    # D = x[1:N_x]  
    # beta = x[N_x+1:2*N_x]
    # error = M*x-log_ratio
    x1 = log_ratio[1:N_x]
    x2 = log_ratio[N_x+1:2*N_x]
    c = log_ratio[end]
    D = (x1+x2.-2*c)/alpha^2
    beta = (x1-x2)/(2*alpha)
    error = 0
    return D, beta, error
end

""" func_Phi: the potential function, i.e the posterior is proportional to exp( - func_Phi)"""
function update_ensemble!(gmgd::BBVIObj{FT, IT}, func_Phi::Function, dt_max::FT, iter::IT, N_iter::IT) where {FT<:AbstractFloat, IT<:Int} #从某一步到下一步的步骤
    
    update_covariance = gmgd.update_covariance
    sqrt_matrix_type = gmgd.sqrt_matrix_type
    diagonal_covariance = gmgd.diagonal_covariance
    discretize_inv_covariance = gmgd.discretize_inv_covariance


    gmgd.iter += 1
    N_x,  N_modes = gmgd.N_x, gmgd.N_modes

    x_mean  = gmgd.x_mean[end]
    logx_w  = gmgd.logx_w[end]
    xx_cov  = gmgd.xx_cov[end]

    sqrt_xx_cov, inv_sqrt_xx_cov = [], []
    for im = 1:N_modes
        sqrt_cov, inv_sqrt_cov = compute_sqrt_matrix(xx_cov[im,:,:]; type=sqrt_matrix_type) 
        Q = random_orthogonal_matrix(N_x)
        push!(sqrt_xx_cov, sqrt_cov*Q)
        push!(inv_sqrt_xx_cov, Q'*inv_sqrt_cov) 
    end

    N_ens = gmgd.N_ens
    d_logx_w, d_x_mean, d_xx_cov = zeros(N_modes), zeros(N_modes, N_x), zeros(N_modes, N_x, N_x)

    D_norm = []

    for im = 1:N_modes 

        # x_p = construct_ensemble(x_mean[im,:], sqrt_xx_cov[im]; c_weights = nothing, N_ens = N_ens)

        alpha = 0.01
        x_p = hcat(sqrt_xx_cov[im], -sqrt_xx_cov[im], zeros(N_x))'*alpha + ones(N_ens)*x_mean[im,:]'

        # log_ratio[i] = logρ[x_p[i,:]] + log func_Phi[x_p[i,:]]
        log_ratio = zeros(N_ens) 
        for i = 1:N_ens
            for imm = 1:N_modes
                log_ratio[i] += exp(logx_w[imm])*Gaussian_density_helper(x_mean[imm,:], inv_sqrt_xx_cov[imm], x_p[i,:])
            end
            log_ratio[i] = log(log_ratio[i])+func_Phi(x_p[i,:])
        end

        c = log_ratio[end]
        D = (log_ratio[1:N_x]+log_ratio[N_x+1:2*N_x].-2*c)/alpha^2
        beta = (log_ratio[1:N_x]-log_ratio[N_x+1:2*N_x])/(2*alpha)
        
        log_ratio_m1 = sqrt_xx_cov[im]*beta
        log_ratio_m2 = sqrt_xx_cov[im].*(D')*sqrt_xx_cov[im]'

        d_x_mean[im,:] = -log_ratio_m1
        d_xx_cov[im,:,:] = -log_ratio_m2
        d_logx_w[im] = -c-sum(D)/2

        push!(D_norm, maximum(abs.(D)))

    end
    
    x_mean_n = copy(x_mean) 
    xx_cov_n = copy(xx_cov)
    logx_w_n = copy(logx_w)

    matrix_norm, vector_norm = [], []
    for im = 1 : N_modes
        # push!(matrix_norm, opnorm( inv_sqrt_xx_cov[im]*d_xx_cov[im,:,:]*inv_sqrt_xx_cov[im]', 2))
        push!(vector_norm, norm(d_x_mean[im,:])/(norm(x_mean[im,:]) + 0.01))
    end
    
    # # set an upper bound dt_max, with cos annealing
    lower_bound = 0.25
    annealing_rate = (lower_bound + (1 - lower_bound)*cos(pi/2 * gmgd.iter/N_iter))
    dt = min(dt_max, annealing_rate/maximum(D_norm)) # keep the matrix postive definite, avoid too large cov update.
    
    if gmgd.iter%20==0
        # @info "dt, |dm|, |dC|, annealing_dt, |C| = ", dt, norm(d_x_mean), norm(d_xx_cov), (0.01 + (1.0 - 0.01)*cos(pi/2 * iter/N_iter)), maximum(matrix_norm) 
        @info "dt=", dt
    end 

    if update_covariance
        
        for im =1:N_modes
            if discretize_inv_covariance
                xx_cov_n[im,:,:] = xx_cov[im,:,:]*inv(I-dt*inv_sqrt_xx_cov[im]'*inv_sqrt_xx_cov[im]*d_xx_cov[im,:,:])
            else
                xx_cov_n[im,:,:] += dt*d_xx_cov[im,:,:]
            end
            xx_cov_n[im, :, :] = Hermitian(xx_cov_n[im, :, :])
            if diagonal_covariance
                xx_cov_n[im, :, :] = diagm(diag(xx_cov_n[im, :, :]))
            end
            if !isposdef(Hermitian(xx_cov_n[im, :, :]))
                @show gmgd.iter
                @info "error! negative determinant for mode ", im,  x_mean[im, :], xx_cov[im, :, :], inv(xx_cov[im, :, :])
                @assert(isposdef(xx_cov_n[im, :, :]))
            end
        end
    end
    # for im =1:N_modes
    #     x_mean_n[im,:] += dt * xx_cov_n[im,:,:]\(xx_cov[im,:,:]*d_x_mean[im,:])
    # end
    x_mean_n += dt * d_x_mean 
    logx_w_n += dt * d_logx_w

    # Normalization
    w_min = gmgd.w_min
    logx_w_n .-= maximum(logx_w_n)
    logx_w_n .-= log( sum(exp.(logx_w_n)) )
    x_w_n = exp.(logx_w_n)
    clip_ind = x_w_n .< w_min
    x_w_n[clip_ind] .= w_min
    x_w_n[(!).(clip_ind)] /= (1 - sum(clip_ind)*w_min)/sum(x_w_n[(!).(clip_ind)])
    logx_w_n .= log.(x_w_n)
    
    
    ######### Save results
    push!(gmgd.x_mean, x_mean_n)
    push!(gmgd.xx_cov, xx_cov_n)
    push!(gmgd.logx_w, logx_w_n) 
end


##########
function Gaussian_mixture_BBVI(func_Phi, x0_w, x0_mean, xx0_cov;
     diagonal_covariance::Bool = false, discretize_inv_covariance::Bool = true, N_iter = 100, dt = 5.0e-1, N_ens = -1)

    _, N_x = size(x0_mean) 
    if N_ens == -1 
        N_ens = 5*N_x
    end

    gmgdobj=BBVIObj(
        x0_w, x0_mean, xx0_cov;
        update_covariance = true,
        diagonal_covariance = diagonal_covariance,
        discretize_inv_covariance = discretize_inv_covariance,
        sqrt_matrix_type = "Cholesky",
        N_ens = N_ens,
        w_min = 1.0e-8)

    for i in 1:N_iter
        if i%max(1, div(N_iter, 10)) == 0  @info "iter = ", i, " / ", N_iter  end
        
        update_ensemble!(gmgdobj, func_Phi, dt,  i,  N_iter) 
    end
    
    return gmgdobj
end
