using Random
using PyPlot
using Distributions
using LinearAlgebra
using ForwardDiff
using DocStringExtensions



# File_01_QuadratureRule
function generate_quadrature_rule(N_x, quadrature_type; c_weight=sqrt(N_x), N_ens = -1)
    if quadrature_type == "mean_point"
        N_ens = 1
        c_weights    = zeros(N_x, N_ens) #√C的系数,列向量
        mean_weights = ones(N_ens) #w_i
    elseif quadrature_type == "random_sampling"
        c_weights = nothing
        mean_weights =  nothing   
    elseif  quadrature_type == "unscented_transform"
        N_ens = 2N_x+1
        c_weights = zeros(N_x, N_ens)
        for i = 1:N_x
            c_weights[i, i+1]      =  c_weight
            c_weights[i, N_x+i+1]  = -c_weight
        end
        mean_weights = fill(1/(2.0*c_weight^2), N_ens)
        # warning: when c_weight <= sqrt(N_x), the weight is negative
        # the optimal is sqrt(3), 
        # Julier, S. J., Uhlmann, J. K., & Durrant-Whyte, H. F. (2000). A new method for nonlinear transformation of means and covariances in filters and estimators. 
        mean_weights[1] = 1 - N_x/c_weight^2
    
    elseif quadrature_type == "cubature_transform_o3"
        N_ens = 2N_x
        c_weight = sqrt(N_x)
        c_weights = zeros(N_x, N_ens)
        for i = 1:N_x
            c_weights[i, i]          =  c_weight
            c_weights[i, N_x+i]      =  -c_weight
        end
        mean_weights = ones(N_ens)/N_ens 

    elseif quadrature_type == "cubature_transform_o5"
        # High-degree cubature Kalman filter
        # Bin Jia, Ming Xin, Yang Cheng
        N_ens = 2N_x*N_x + 1
        c_weights    = zeros(N_x, N_ens)
        mean_weights = ones(N_ens)

        mean_weights[1] = 2.0/(N_x + 2)

        for i = 1:N_x
            c_weights[i, 1+i]          =  sqrt(N_x+2)
            c_weights[i, 1+N_x+i]      =  -sqrt(N_x+2)
            mean_weights[1+i] = mean_weights[1+N_x+i] = (4 - N_x)/(2*(N_x+2)^2)
        end
        ind = div(N_x*(N_x - 1),2)
        for i = 1: N_x
            for j = i+1:N_x
                c_weights[i, 2N_x+1+div((2N_x-i)*(i-1),2)+(j-i)],      c_weights[j, 2N_x+1+div((2N_x-i)*(i-1),2)+(j-i)]        =   sqrt(N_x/2+1),  sqrt(N_x/2+1)
                c_weights[i, 2N_x+1+ind+div((2N_x-i)*(i-1),2)+(j-i)],  c_weights[j, 2N_x+ind+1+div((2N_x-i)*(i-1),2)+(j-i)]    =  -sqrt(N_x/2+1), -sqrt(N_x/2+1)
                c_weights[i, 2N_x+1+2ind+div((2N_x-i)*(i-1),2)+(j-i)], c_weights[j, 2N_x+2ind+1+div((2N_x-i)*(i-1),2)+(j-i)]   =   sqrt(N_x/2+1), -sqrt(N_x/2+1)
                c_weights[i, 2N_x+1+3ind+div((2N_x-i)*(i-1),2)+(j-i)], c_weights[j, 2N_x+3ind+1+div((2N_x-i)*(i-1),2)+(j-i)]   =  -sqrt(N_x/2+1),  sqrt(N_x/2+1)
                
                mean_weights[2N_x+1+div((2N_x-i)*(i-1),2)+(j-i)]      = 1.0/(N_x+2)^2
                mean_weights[2N_x+1+ind+div((2N_x-i)*(i-1),2)+(j-i)]  = 1.0/(N_x+2)^2
                mean_weights[2N_x+1+2ind+div((2N_x-i)*(i-1),2)+(j-i)] = 1.0/(N_x+2)^2
                mean_weights[2N_x+1+3ind+div((2N_x-i)*(i-1),2)+(j-i)] = 1.0/(N_x+2)^2
            end
        end

    else 
        print("cubature tansform with quadrature type ", quadrature_type, " has not implemented.")

    end

    return N_ens, c_weights, mean_weights
end

function compute_sqrt_matrix(C; type="Cholesky")
    if type == "Cholesky"
        sqrt_cov, inv_sqrt_cov = cholesky(Hermitian(C)).L,  inv(cholesky(Hermitian(C)).L) 
    elseif type == "SVD"
        U, D, _ = svd(Hermitian(C))
        sqrt_cov, inv_sqrt_cov = U*Diagonal(sqrt.(D)),  Diagonal(sqrt.(1.0./D))*U' 
        
    else
        print("Type ", type, " for computing sqrt matrix has not implemented.")
    end
    return sqrt_cov, inv_sqrt_cov
end

function construct_ensemble(x_mean, sqrt_cov; c_weights = nothing, N_ens = 10) #生成点θ_i

    N_x = size(x_mean)

    if c_weights === nothing
        xs = ones(N_ens)*x_mean' + (sqrt_cov * rand(Normal(0, 1),N_x, N_ens))'
    else
        N_ens = size(c_weights,2)
        xs = ones(N_ens)*x_mean' + (sqrt_cov * c_weights)'
    end

    return xs
end

# Derivative Free
function compute_expectation_BIP(x_mean, inv_sqrt_cov, V, c_weight) #V是F算出的生成点θ_i处值

    N_x = length(x_mean)
    Φᵣ_mean, ∇Φᵣ_mean, ∇²Φᵣ_mean = 0.0, zeros(N_x), zeros(N_x, N_x)

    
    α = c_weight
    N_ens, N_f = size(V)
    a = zeros(N_x, N_f)
    b = zeros(N_x, N_f)
    c = zeros(N_f)
    
    c = V[1, :]
    for i = 1:N_x
        a[i, :] = (V[i+1, :] + V[i+N_x+1, :] - 2*V[1, :])/(2*α^2)
        b[i, :] = (V[i+1, :] - V[i+N_x+1, :])/(2*α)
    end
    ATA = a * a'
    BTB = b * b'
    BTA = b * a'
    BTc, ATc = b * c, a * c
    cTc = c' * c
    # Φᵣ_mean = 1/2*(sum(ATA) + 2*tr(ATA) + 2*sum(ATc) + tr(BTB) + cTc)
    Φᵣ_mean = 1/2 * (cTc)
    # ∇Φᵣ_mean = inv_sqrt_cov'*(sum(BTA,dims=2) + 2*diag(BTA) + BTc)
    # ∇Φᵣ_mean = inv_sqrt_cov'*(3*diag(BTA) + BTc)
    # Ignore second order effect
    ∇Φᵣ_mean = inv_sqrt_cov'*(BTc)
    # ∇²Φᵣ_mean = inv_sqrt_cov'*( Diagonal(2*dropdims(sum(ATA, dims=2), dims=2) + 4*diag(ATA) + 2*ATc) + BTB)*inv_sqrt_cov
    # ∇²Φᵣ_mean = inv_sqrt_cov'*(Diagonal(6*diag(ATA)) + BTB)*inv_sqrt_cov
    # ∇²Φᵣ_mean = inv_sqrt_cov'*( Diagonal(2*fill(sum(ATA)/N_x, N_x) + 4*diag(ATA) + 2*ATc) + BTB)*inv_sqrt_cov
    
    ∇²Φᵣ_mean = inv_sqrt_cov'*( Diagonal(6*diag(ATA) + 2*ATc) + BTB)*inv_sqrt_cov 
             
    return Φᵣ_mean, ∇Φᵣ_mean, ∇²Φᵣ_mean
end


function compute_expectation(V, ∇V, ∇²V, mean_weights)

    N_ens, N_x = size(∇V)
    Φᵣ_mean, ∇Φᵣ_mean, ∇²Φᵣ_mean = 0.0, zeros(N_x), zeros(N_x, N_x)

    # general sampling problem Φᵣ = V 
    Φᵣ_mean   = mean_weights' * V
    ∇Φᵣ_mean  = ∇V' * mean_weights
    ∇²Φᵣ_mean = sum(mean_weights[i] * ∇²V[i, :, :] for i = 1:length(mean_weights))

    return Φᵣ_mean, ∇Φᵣ_mean, ∇²Φᵣ_mean
end

#F₁ = xᵀA₁x + b₁ᵀx₁ + c₁
#F₂ = xᵀA₂x + b₂ᵀx₂ + c₂
#Φᵣ = FᵀF/2

function func_F(x, args)
    A₁,b₁,c₁,A₂,b₂,c₂ = args
    return [x'*A₁*x + b₁'*x + c₁; 
            x'*A₂*x + b₂'*x + c₂]
end

function func_Phi_R(x, args)
    F = func_F(x, args)
    Φᵣ = (F' * F)/2.0
    return Φᵣ
end

function func_dF(x, args)
    return func_F(x, args), 
           ForwardDiff.jacobian(x -> func_F(x, args), x)
end

function func_dPhi_R(x, args)
    return func_Phi_R(x, args), 
           ForwardDiff.gradient(x -> func_Phi_R(x, args), x), 
           ForwardDiff.hessian(x -> func_Phi_R(x, args), x)
end


#File_02_GMGD
mutable struct GMGDObj{FT<:AbstractFloat, IT<:Int}
    "object name"
    name::String
    "a vector of arrays of size (N_modes) containing the modal weights of the parameters"
    logx_w::Vector{Array{FT, 1}} #FT是类型，每一步是一个Array{FT, 1}，存的是logwi
    "a vector of arrays of size (N_modes x N_parameters) containing the modal means of the parameters"
    x_mean::Vector{Array{FT, 2}} #每一步期望
    "a vector of arrays of size (N_modes x N_parameters x N_parameters) containing the modal covariances of the parameters"
    xx_cov::Union{Vector{Array{FT, 3}}, Nothing} #每一步的cov
    "number of modes"
    N_modes::IT
    "size of x"
    N_x::IT
    "current iteration number"
    iter::IT
    "update covariance or not"
    update_covariance::Bool
    "Cholesky, SVD"
    sqrt_matrix_type::String
    "expectation of Gaussian mixture and its derivatives"
    quadrature_type_GM::String
    c_weight_GM::FT
    c_weights_GM::Array{FT, 2}
    mean_weights_GM::Array{FT, 1}
    "when Bayesian_inverse_problem is true :  function is F, 
     otherwise the function is Phi_R,  Phi_R = 1/2 F ⋅ F"
    Bayesian_inverse_problem::Bool
    "Bayesian inverse problem observation dimension"
    N_f::IT
    "sample points"
    N_ens::IT
    "quadrature points for expectation, 
     random_sampling,  mean_point,  unscented_transform"
    quadrature_type::String
    "derivative_free: 0, first_order: 1, second_order: 2"
    gradient_computation_order::Int64
    "expectation of Gaussian mixture and its derivatives"
    c_weight_BIP::FT
    c_weights::Array{FT, 2}
    mean_weights::Array{FT, 1}
    "weight clipping"
    w_min::FT

    
end


function GMGDObj(# initial condition  #初始化这个struct类
                x0_w::Array{FT, 1},
                x0_mean::Array{FT, 2},
                xx0_cov::Union{Array{FT, 3}, Nothing};
                update_covariance::Bool = true,
                sqrt_matrix_type::String = "Cholesky",
                # setup for Gaussian mixture part
                quadrature_type_GM::String = "cubature_transform_o5",
                c_weight_GM::FT = sqrt(3.0),
                # setup for potential function part
                Bayesian_inverse_problem::Bool = false,
                N_f::IT = 1,
                gradient_computation_order::IT = 2, 
                quadrature_type = "unscented_transform",
                c_weight_BIP::FT = sqrt(3.0),
                N_ens::IT = -1,
                w_min::FT = 1.0e-15) where {FT<:AbstractFloat, IT<:Int}

    N_modes, N_x = size(x0_mean)

    logx_w = Array{FT,1}[]      # array of Array{FT, 1}'s
    push!(logx_w, log.(x0_w))   # insert parameters at end of array (in this case just 1st entry)
    x_mean = Array{FT,2}[]      # array of Array{FT, 2}'s
    push!(x_mean, x0_mean)      # insert parameters at end of array (in this case just 1st entry)
    xx_cov = Array{FT,3}[]      # array of Array{FT, 2}'s
    push!(xx_cov, xx0_cov)      # insert parameters at end of array (in this case just 1st entry)
    
    iter = 0
    _, c_weights_GM, mean_weights_GM = generate_quadrature_rule(N_x, quadrature_type_GM; c_weight=c_weight_GM)
    
    N_ens, c_weights, mean_weights = generate_quadrature_rule(N_x, quadrature_type; c_weight=c_weight_BIP, N_ens=N_ens)
    
     
    name = (Bayesian_inverse_problem ? "Derivative free GMGD" : "GMGD")
    GMGDObj(name,
            logx_w, x_mean, xx_cov, N_modes, N_x,
            iter, update_covariance,
            sqrt_matrix_type,
            ## Gaussian mixture expectation
            quadrature_type_GM, c_weight_GM, c_weights_GM, mean_weights_GM,
            ## potential function expectation
            Bayesian_inverse_problem, N_f, N_ens, quadrature_type, gradient_computation_order,
            c_weight_BIP, c_weights, mean_weights, w_min)
end

# avoid computing 1/(2π^N_x/2)
function Gaussian_density_helper(x_mean::Array{FT,1}, inv_sqrt_xx_cov, x::Array{FT,1}) where {FT<:AbstractFloat}
    return exp( -1/2*((x - x_mean)'* (inv_sqrt_xx_cov'*inv_sqrt_xx_cov*(x - x_mean)) )) * abs(det(inv_sqrt_xx_cov))
end


# avoid computing 1/(2π^N_x/2) for ρ, ∇ρ, ∇²ρ
function Gaussian_mixture_density_derivatives(x_w::Array{FT,1}, x_mean::Array{FT,2}, inv_sqrt_xx_cov, x::Array{FT,1}; hessian_correct::Bool = false) where {FT<:AbstractFloat}
    N_modes, N_x = size(x_mean)

    ρ = 0.0
    ∇ρ = zeros(N_x)
    ∇²ρ = zeros(N_x, N_x)
   
    for i = 1:N_modes
        temp = inv_sqrt_xx_cov[i]'*inv_sqrt_xx_cov[i]*(x_mean[i,:] - x)
        ρᵢ   = Gaussian_density_helper(x_mean[i,:], inv_sqrt_xx_cov[i], x)
        # TODO compute ratio
        ρ   += x_w[i]*ρᵢ
        ∇ρ  += x_w[i]*ρᵢ*temp
        ∇²ρ += (hessian_correct ? x_w[i]*ρᵢ*( temp * temp') : x_w[i]*ρᵢ*( temp * temp' - inv_sqrt_xx_cov[i]'*inv_sqrt_xx_cov[i]))
    end

    return ρ, ∇ρ, ∇²ρ
end


function compute_logρ_gm(x_p, x_w, x_mean, inv_sqrt_xx_cov; hessian_correct::Bool = false) #给定n个modes，每个modes有N_ens个点，求出logρ导数和二阶导数
    N_modes, N_ens, N_x = size(x_p) #x_p是三维矩阵 N_modes -> K
    logρ = zeros(N_modes, N_ens)
    ∇logρ = zeros(N_modes, N_ens, N_x)
    ∇²logρ = zeros(N_modes, N_ens, N_x, N_x)
    for im = 1:N_modes
        for i = 1:N_ens
            # ρ, ∇ρ, ∇²ρ = Gaussian_mixture_density_derivatives(x_w, x_mean, inv_sqrt_xx_cov, x_p[im, i, :]; hessian_correct = hessian_correct)
            ρ, ∇ρ  , ∇²ρ = Gaussian_mixture_density_derivatives(x_w, x_mean, inv_sqrt_xx_cov, x_p[im, i, :]; hessian_correct = hessian_correct)
            
            logρ[im, i]         =   log(  ρ  ) - N_x/2.0 * log(2π)
            ∇logρ[im, i, :]     =   ∇ρ/ρ
            ∇²logρ[im, i, :, :] =  (hessian_correct ? ∇²ρ/ρ - (∇ρ/ρ^2)*∇ρ' - inv_sqrt_xx_cov[im]'*inv_sqrt_xx_cov[im] : (∇²ρ/ρ - (∇ρ/ρ^2)*∇ρ'))
        end
    end

    return logρ, ∇logρ, ∇²logρ
end



#需要更改的地方！
function compute_logρ_gm_expectation(x_w, x_mean, sqrt_xx_cov, inv_sqrt_xx_cov, c_weights_GM, mean_weights_GM; hessian_correct::Bool = false)
    x_w = x_w / sum(x_w)
    N_modes, N_x = size(x_mean)
    _, N_ens = size(c_weights_GM)
    xs = zeros(N_modes, N_ens, N_x)
    logρ_mean, ∇logρ_mean, ∇²logρ_mean = zeros(N_modes), zeros(N_modes, N_x), zeros(N_modes, N_x, N_x)

    for im = 1:N_modes
        xs[im,:,:] = construct_ensemble(x_mean[im, :], sqrt_xx_cov[im]; c_weights = c_weights_GM)
    end
    
    logρ, ∇logρ, ∇²logρ = compute_logρ_gm(xs, x_w, x_mean, inv_sqrt_xx_cov; hessian_correct = hessian_correct)
   
    for im = 1:N_modes
        logρ_mean[im], ∇logρ_mean[im,:], ∇²logρ_mean[im,:,:] = compute_expectation(logρ[im,:], ∇logρ[im,:,:], ∇²logρ[im,:,:,:], mean_weights_GM)
    end

    return  logρ_mean, ∇logρ_mean, ∇²logρ_mean
end
   

function update_ensemble!(gmgd::GMGDObj{FT, IT}, func::Function, dt::FT, time::IT) where {FT<:AbstractFloat, IT<:Int} #从某一步到下一步的步骤
    
    update_covariance = gmgd.update_covariance
    sqrt_matrix_type = gmgd.sqrt_matrix_type

    gmgd.iter += 1
    N_x,  N_modes = gmgd.N_x, gmgd.N_modes

    x_mean  = gmgd.x_mean[end]
    logx_w  = gmgd.logx_w[end]
    xx_cov  = gmgd.xx_cov[end]

    sqrt_xx_cov, inv_sqrt_xx_cov = [], []
    for im = 1:N_modes
        sqrt_cov, inv_sqrt_cov = compute_sqrt_matrix(xx_cov[im,:,:]; type=sqrt_matrix_type) 
        push!(sqrt_xx_cov, sqrt_cov)
        push!(inv_sqrt_xx_cov, inv_sqrt_cov) 
    end

    ###########  Entropy term
    N_ens, c_weights_GM, mean_weights_GM = gmgd.N_ens, gmgd.c_weights_GM, gmgd.mean_weights_GM

    # 注意：以下一行有计算浪费！无需计算出∇²logρ_mean！
    logρ_mean, ∇logρ_mean, _ = compute_logρ_gm_expectation(exp.(logx_w), x_mean, sqrt_xx_cov, inv_sqrt_xx_cov, c_weights_GM, mean_weights_GM; hessian_correct=true)
    ∇²logρ_mean = zeros(N_modes, N_x, N_x)
    for dim = 1 : N_x
        sqrt_xx_cov_dim, inv_sqrt_xx_cov_dim = [], []
        for im = 1:N_modes
            sqrt_cov_dim, inv_sqrt_cov_dim = compute_sqrt_matrix(xx_cov[im,dim:dim,dim:dim]; type=sqrt_matrix_type) 
            push!(sqrt_xx_cov_dim, sqrt_cov_dim)
            push!(inv_sqrt_xx_cov_dim, inv_sqrt_cov_dim) 
        end
        c_weights_GM_dim = zeros(1, size(c_weights_GM,2)) #每一个维度的c_weights 
        c_weights_GM_dim[1,:] = c_weights_GM[dim,:]
        
        x_w = exp.(logx_w) / sum(exp.(logx_w))
        xs = zeros(size(x_mean, 1), size(c_weights_GM, 2), size(x_mean, 2))
        for im = 1:N_modes
            xs[im,:,:] = construct_ensemble(x_mean[im, :], sqrt_xx_cov[im]; c_weights = c_weights_GM)
        end
        
        logρ, ∇logρ, ∇²logρ = compute_logρ_gm(xs, x_w, x_mean, inv_sqrt_xx_cov; hessian_correct = true)
    
        # 对角部分重新装填∇²logρ_mean
        for im = 1:N_modes
            _,_,∇²logρ_mean[im,dim:dim,dim:dim] = compute_expectation(logρ[im,:], ∇logρ[im,:,dim:dim], ∇²logρ[im,:,dim:dim,dim:dim], mean_weights_GM)
        end

    end

    # if isnan.(norm(∇²logρ_mean))
    #     @info ∇²logρ_mean
    # end
    # @info norm(logρ_mean - logρ_mean_), norm(∇logρ_mean - ∇logρ_mean_) , norm(∇²logρ_mean), norm(∇²logρ_mean_)
    # @assert(norm(logρ_mean - logρ_mean_)+norm(∇logρ_mean - ∇logρ_mean_)+norm(∇²logρ_mean - ∇²logρ_mean_)<1.0e-8)
    ############ Generate sigma points
    x_p = zeros(N_modes, N_ens, N_x)
    for im = 1:N_modes
        x_p[im,:,:] = construct_ensemble(x_mean[im,:], sqrt_xx_cov[im]; c_weights = gmgd.c_weights) #采样的点
    end
    ###########  Potential term

    V, ∇V, ∇²V = func(x_p)

    Φᵣ_mean, ∇Φᵣ_mean, ∇²Φᵣ_mean = zeros(N_modes), zeros(N_modes, N_x), zeros(N_modes, N_x, N_x)
    for im = 1:N_modes
        Φᵣ_mean[im], ∇Φᵣ_mean[im,:], ∇²Φᵣ_mean[im,:,:] = gmgd.Bayesian_inverse_problem ?   #算出Φ的期望
        compute_expectation_BIP(x_mean[im,:], inv_sqrt_xx_cov[im], V[im,:,:], gmgd.c_weight_BIP) : 
        compute_expectation(V[im,:], ∇V[im,:,:], ∇²V[im,:,:,:], gmgd.mean_weights) 
    end

    x_mean_n = copy(x_mean) #下一个时刻的，copy过来然后修改
    xx_cov_n = copy(xx_cov)
    logx_w_n = copy(logx_w)

    m_ens_mean = zeros(N_modes, N_x)
    C_ens_mean = zeros(N_modes, N_x, N_x)
    m_ens_value = zeros(N_modes, N_ens, N_x)
    C_ens_value = zeros(N_modes, N_ens, N_x, N_x)

    x_w = zeros(size(logx_w)[1])
    for im = 1 : N_modes
        for i = 1 : size(logx_w)[1]
            x_w[i] = exp(logx_w[i])
        end
        # z = rand(MvNormal(x_mean[im, :], xx_cov[im, :, :]))
        # ρ, ∇ρ , ∇²ρ = Gaussian_mixture_density_derivatives(x_w, x_mean, inv_sqrt_xx_cov, z; hessian_correct = true)
        # wk = (Gaussian_density_helper(x_mean[im,:], inv_sqrt_xx_cov[im], z)) / (ρ)
        # wk = wk * exp(logx_w[im])

        #计算残余项
        #进行m的ensemble
        function m_ens(im, x)
            return x_w[im] * (Gaussian_density_helper(x_mean[im,:], inv_sqrt_xx_cov[im], x) * inv(xx_cov[im,:,:]) * (x - x_mean[im,:])) / Gaussian_mixture_density_derivatives(x_w, x_mean, inv_sqrt_xx_cov, x; hessian_correct = true)[1]
        end

        for i = 1:N_ens
            m_ens_value[im, i, :]= m_ens(im, x_p[im, i, :])
        end
        
        #进行C的ensemble
        function C_ens(im, x)
            #return 2 * xx_cov[im, :, :] ^2 * x_w[im] * (0.5 * Gaussian_density_helper(x_mean[im,:], inv_sqrt_xx_cov[im], x)) * (xx_cov[im,:,:] - (x - x_mean[im,:]) * (x - x_mean[im,:])' ) / Gaussian_mixture_density_derivatives(x_w, x_mean, inv_sqrt_xx_cov, x; hessian_correct = true)[1]
             return x_w[im] * (0.5 * Gaussian_density_helper(x_mean[im,:], inv_sqrt_xx_cov[im], x)) * (xx_cov[im,:,:] - (x - x_mean[im,:]) * (x - x_mean[im,:])' ) / Gaussian_mixture_density_derivatives(x_w, x_mean, inv_sqrt_xx_cov, x; hessian_correct = true)[1]
        end

        for i = 1:N_ens
            C_ens_value[im, i, :, :]= C_ens(im, x_p[im, i, :])
        end
        #计算m, C残余项均值
        _, m_ens_mean[im, :], C_ens_mean[im, :, :] = compute_expectation(0, m_ens_value[im,:,:], C_ens_value[im,:,:,:], gmgd.mean_weights)
    end

    #开始更新

    

    dt_upper1 = 1.0
    dt_upper_limit = 0.5
    array_norm = []
    for i = 1 : N_modes
        push!(array_norm, norm(xx_cov[i,:,:] * (∇²logρ_mean[i, :, :] + ∇²Φᵣ_mean[i, :, :] - 2 * inv(xx_cov[i,:,:]) * C_ens_mean[i, :, :] * inv(xx_cov[i,:,:])), Inf))
    end
    dt = min(dt_upper1 - (2 * (dt_upper1 - dt_upper_limit) * atan(π * time / 180)) / π, 0.99 / (maximum(array_norm))) # 保证矩阵正定
    #dt = 1e-2
    
    for im = 1:N_modes
         

        if update_covariance
            tempim = xx_cov[im, :, :]
            for ii = 1 : size(tempim,1)
                for jj = 1 : size(tempim,2)
                    if ii == jj
                        tempim[ii,jj] = ∇²logρ_mean[im, ii, jj] + ∇²Φᵣ_mean[im, ii, jj] # (C_ens_mean) * (2 * inv(xx_cov[im, :, :])[ii, jj] ^ 2) # 
                    else
                        tempim[ii,jj] = 0
                    end
                end
            end
            # tempic = C_ens_mean[im, :, :]
            # for ii = 1 : size(tempim,1)
            #     for jj = 1 : size(tempim,2)
            #         if ii == jj
            #             tempic[ii,jj] = 2 * tempic[ii,jj] * inv(xx_cov[im, ii, jj])^2 # (C_ens_mean) * (2 * inv(xx_cov[im, :, :])[ii, jj] ^ 2) # 
            #         else
            #             tempic[ii,jj] = 0
            #         end
            #     end
            # end
            # tempic = inv(xx_cov[im, :, :]) * C_ens_mean[im, :, :] * 2 * inv(xx_cov[im, :, :])
            
            # loginvxx_cov = log.(inv(xx_cov[im, :, :]))
            # loginvxx_cov_n =  loginvxx_cov + dt * (tempim - tempic)
            # xx_cov_n[im, :, :] = inv(exp.(loginvxx_cov_n))
            # xx_cov_n[im, :, :] =  inv( inv(xx_cov[im, :, :]) + dt*(∇²logρ_mean[im, :, :] + ∇²Φᵣ_mean[im, :, :] - tempic) )
             xx_cov_n[im, :, :] = inv(inv(xx_cov[im, :, :]) + dt * (tempim )) #- tempic
            #xx_cov_n[im, :, :] =  inv( inv(xx_cov[im, :, :]) + dt * (tempim + wk * 0.5 * (z - x_mean[im, :]) * (z - x_mean[im, :])'- wk * xx_cov[im, :, :]) ) #
            #xx_cov_n[im, :, :] =  inv( inv(xx_cov[im, :, :]) + dt*(∇²logρ_mean[im, :, :] + ∇²Φᵣ_mean[im, :, :]) )
            # xx_cov_n[im, :, :] =  (1 + dt)*inv( inv(xx_cov[im, :, :]) + dt*(∇²logρ_mean[im, :, :] + ∇²V_mean[im, :, :]) )
            # @info "cov residual ", ∇²logρ_mean[im, :, :], ∇²Φᵣ_mean[im, :, :], ∇²logρ_mean[im, :, :] + ∇²Φᵣ_mean[im, :, :]
        
            if det(xx_cov_n[im, :, :]) <= 0.0
                @info "error! negative determinant for mode ", im,  x_mean[im, :], xx_cov[im, :, :], inv(xx_cov[im, :, :]), ∇²logρ_mean[im, :, :], ∇²Φᵣ_mean[im, :, :]
                @info " mean residual ", ∇logρ_mean[im, :] , ∇Φᵣ_mean[im, :], ∇logρ_mean[im, :] + ∇Φᵣ_mean[im, :]
            end
            
        else
            xx_cov_n[im, :, :] = xx_cov[im, :, :]
        end

        
            # temp = inv_sqrt_xx_cov[im]'*inv_sqrt_xx_cov[im]*(x_mean[im,:] - z)
            # ρᵢ   = Gaussian_density_helper(x_mean[im,:], inv(xx_cov[im,:,:]), z)
            # ∇ρᵢ  = exp(logx_w[im])*ρᵢ*temp
         
        
        tempimm = ∇logρ_mean[im, :] + ∇Φᵣ_mean[im, :] #- m_ens_mean[im,:]
        tempimm = xx_cov_n[im,:,:] * tempimm
        # for ii = 1 : size(tempimm,1)
        #     tempimm[ii] = tempimm[ii] * xx_cov_n[im,ii,ii]
        # end

        #ρ,_,_ =Gaussian_mixture_density_derivatives(x_w, x_mean, inv_sqrt_xx_cov, z)
        #logρ=log(ρ)
         x_mean_n[im, :]  =  x_mean[im, :] - dt*(tempimm)
        # x_mean_n[im, :]  =  x_mean[im, :] - dt*xx_cov_n[im, :, :]*(∇logρ_mean[im, :] + ∇Φᵣ_mean[im, :] )
        #x_cov和ai待完成!
        
        # 更新权重
        ρlogρ_Φᵣ = 0 
        for jm = 1:N_modes
            ρlogρ_Φᵣ += exp(logx_w[jm])*(logρ_mean[jm] + Φᵣ_mean[jm])
        end
        logx_w_n[im] = logx_w[im] - dt*(logρ_mean[im] + Φᵣ_mean[im]) #- ρlogρ_Φᵣ

        # @info "weight residual ", logρ_mean[im], Φᵣ_mean[im], ρlogρ_Φᵣ, logρ_mean[im] + Φᵣ_mean[im] - ρlogρ_Φᵣ
        
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
    
    
    ########### Save resutls
    push!(gmgd.x_mean, x_mean_n)   # N_ens x N_params
    push!(gmgd.xx_cov, xx_cov_n)   # N_ens x N_data
    push!(gmgd.logx_w, logx_w_n)   # N_ens x N_data
end

function ensemble(x_ens, forward) #forward是一个函数，存下这个函数的导数，一阶导数，二阶导数，算的是Φᵣ
    N_modes, N_ens, N_x = size(x_ens)

    V = zeros(N_modes, N_ens)   
    ∇V = zeros(N_modes, N_ens, N_x)   
    ∇²V = zeros(N_modes, N_ens, N_x, N_x)  

    for im = 1:N_modes
        for i = 1:N_ens
            V[im, i], ∇V[im, i, :], ∇²V[im, i, :, :] = forward(x_ens[im, i, :])
        end
    end

    return V, ∇V, ∇²V 
end


function ensemble_BIP(x_ens, forward, N_f) #算F
    N_modes, N_ens, N_x = size(x_ens)
    V = zeros(N_modes, N_ens, N_f)   
    for im = 1:N_modes
        for i = 1:N_ens
            V[im, i, :] = forward(x_ens[im, i, :])
        end
    end
    
    return V, nothing, nothing
end

function GMGD_Run(
    forward::Function, 
    T::FT, #总共跑的时间
    N_iter::IT, #迭代
    # Initial condition初始条件
    x0_w::Array{FT, 1}, x0_mean::Array{FT, 2}, xx0_cov::Array{FT, 3}; #分号后自己选计算的方式
    update_covariance::Bool = true, 
    sqrt_matrix_type::String = "Cholesky",
    # setup for Gaussian mixture part
    quadrature_type_GM::String = "cubature_transform_o5",
    c_weight_GM::FT = sqrt(3.0),
    # setup for potential function part
    Bayesian_inverse_problem::Bool = false,
    N_f::IT = 1, #f函数的维度
    gradient_computation_order::IT = 2, 
    quadrature_type = "unscented_transform",
    c_weight_BIP::FT = sqrt(3.0), #输入α
    N_ens::IT = -1,
    w_min::FT = 1.0e-15) where {FT<:AbstractFloat, IT<:Int} #最小容忍的w_min
    

    gmgdobj = GMGDObj(# initial condition
        x0_w, x0_mean, xx0_cov;
        update_covariance = update_covariance,
        sqrt_matrix_type = sqrt_matrix_type,
        # setup for Gaussian mixture part
        quadrature_type_GM = quadrature_type_GM,
        c_weight_GM = c_weight_GM,
        # setup for potential function part
        Bayesian_inverse_problem = Bayesian_inverse_problem,
        N_f = N_f,
        gradient_computation_order = gradient_computation_order, 
        quadrature_type = quadrature_type,
        c_weight_BIP = c_weight_BIP,
        N_ens = N_ens,
        w_min = w_min) 

    func(x_ens) = Bayesian_inverse_problem ? ensemble_BIP(x_ens, forward, N_f) : ensemble(x_ens, forward)  
    
    dt = T/N_iter
    for i in 1:N_iter
        if i%div(N_iter, 10) == 0  @info "iter = ", i, " / ", N_iter  end
        
        update_ensemble!(gmgdobj, func, dt, i) 
    end
    
    return gmgdobj
    
end

###### Plot function 


function Gaussian_density_1d(x_mean::Array{FT,1}, inv_sqrt_xx_cov, xx) where {FT<:AbstractFloat}
    dx = [xx[:]' ;] - repeat(x_mean, 1, length(xx))
    return exp.( -1/2*(dx .* (inv_sqrt_xx_cov'*(inv_sqrt_xx_cov*dx)))) .* abs(det(inv_sqrt_xx_cov))
end

function Gaussian_mixture_1d(x_w, x_mean, xx_cov,  xx)
    
    N_modes = length(x_w)
    inv_sqrt_xx_cov = [compute_sqrt_matrix(xx_cov[im,:,:]; type="Cholesky")[2] for im = 1:N_modes]
    
    # 1d Gaussian plot
    dx = xx[2] - xx[1]
    N_x = length(xx)
    y = zeros(N_x)
    
    for im = 1:N_modes
        y .+= x_w[im]*Gaussian_density_1d(x_mean[im,:], inv_sqrt_xx_cov[im], xx)'
    end

    y = y/(sum(y)*dx)
    
    return y 
end


function posterior_BIP_1d(func_F, xx)
    dx = xx[2] - xx[1]
    N_x = length(xx)
    y = zeros(N_x)
    for i = 1:N_x
        F = func_F([xx[i];])
        y[i] = exp(-F'*F/2)
    end
    y /= (sum(y)*dx) 

    return y
end



function posterior_1d(func_V, xx)
    dx = xx[2] - xx[1]
    N_x = length(xx)
    y = zeros(N_x)
    for i = 1:N_x
        V = func_V([xx[i];])
        y[i] = exp(-V)
    end
    y /= (sum(y)*dx) 

    return y
end
    


function visualization_1d(ax; Nx=2000, x_lim=[-4.0,4.0], func_F = nothing, func_V = nothing, objs=nothing)

    # visualization 
    x_min, x_max = x_lim
    
    xx = LinRange(x_min, x_max, Nx)
    dx = xx[2] - xx[1] 
    
    yy_ref = (func_V === nothing ? posterior_BIP_1d(func_F, xx) : posterior_1d(func_V, xx))
    color_lim = (minimum(yy_ref), maximum(yy_ref))
    
    ax[1].plot(xx, yy_ref, "--", label="Reference", color="grey", linewidth=2, fillstyle="none", markevery=25)
    ax[2].plot(xx, yy_ref, "--", label="Reference", color="grey", linewidth=2, fillstyle="none", markevery=25)
           
   
    N_obj = length(objs)
    
    N_iter = length(objs[1].logx_w) - 1
    error = zeros(N_obj, N_iter+1)
        
    for (iobj, obj) in enumerate(objs)
        for iter = 0:N_iter  
            x_w = exp.(obj.logx_w[iter+1]); x_w /= sum(x_w)
            x_mean = obj.x_mean[iter+1]
            xx_cov = obj.xx_cov[iter+1]
            yy = Gaussian_mixture_1d(x_w, x_mean, xx_cov,  xx)
            error[iobj, iter+1] = norm(yy - yy_ref,1)*dx
            
            if iter == N_iter
                ax[iobj].plot(xx, yy, "--", label="Reference", color="red", linewidth=2, fillstyle="none", markevery=25)
                N_modes = size(x_mean, 1)
                

                ax[iobj].scatter(obj.x_mean[1], exp.(obj.logx_w[1]), marker="x", color="grey") 
                ax[iobj].scatter(x_mean, x_w, marker="o", color="red", facecolors="none")

            end
        end
        
    end
    for i_obj = 1:N_obj
        ax[N_obj+1].semilogy(Array(0:N_iter), error[i_obj, :], label=objs[i_obj].name*" (K="*string(size(objs[i_obj].x_mean[1], 1))*")")
    end
    ax[N_obj+1].legend()
end

function visualization_1d_multi(ax; Nx=2000, x_lim=[-4.0,4.0], func_F = nothing, func_V = nothing, objs=nothing, half_objs=nothing, lines = 10)

    # visualization 
    x_min, x_max = x_lim
    
    xx = LinRange(x_min, x_max, Nx)
    dx = xx[2] - xx[1] 
    
    yy_ref = (func_V === nothing ? posterior_BIP_1d(func_F, xx) : posterior_1d(func_V, xx))
    color_lim = (minimum(yy_ref), maximum(yy_ref))
    
    ax[1].plot(xx, yy_ref, "--", label="Reference", color="grey", linewidth=2, fillstyle="none", markevery=25)
    ax[2].plot(xx, yy_ref, "--", label="Reference", color="grey", linewidth=2, fillstyle="none", markevery=25)
           
   
    N_obj = 2
    
    N_iter = length(objs[1].logx_w) - 1
    error = zeros(N_obj, N_iter+1)

    #plot the first picture
    x_w = zeros(length(half_objs[1].logx_w[1]), lines)
    x_mean = zeros(length(half_objs[1].x_mean[1]), lines)
    axx_cov, bxx_cov = size(half_objs[1].xx_cov[1])
    xx_cov = zeros(axx_cov, bxx_cov, lines)
    yy = zeros(length(yy_ref), lines)
    μyy = zeros(length(yy_ref))
    σyy = zeros(length(yy_ref))
    for iter = 0 : N_iter 
        error_mean = 0
        for i = 1 : lines
            x_w[:,i] = exp.(half_objs[i].logx_w[iter+1]); x_w[:,i] /= sum(x_w[:,i])
            x_mean[:,i] = half_objs[i].x_mean[iter+1]
            xx_cov[:,:,i] = half_objs[i].xx_cov[iter+1]
            yy[:,i] = Gaussian_mixture_1d(x_w[:,i], x_mean[:,i], xx_cov[:,:,i],  xx)
            error_mean += norm(yy[:,i] - yy_ref,1)*dx
        end
        error[1, iter+1] = error_mean / lines
        for l = 1 : length(yy_ref)
            μyy[l] = sum(yy[l,:]) / lines
            σyy[l] = norm(yy[l,:] - μyy[l] * ones(lines), 2) / sqrt(lines - 1)
        end
    end
    ax[1].plot(xx, μyy, label="Reference", color="red", linewidth=2, fillstyle="none", markevery=25)
    ax[1].plot(xx, μyy + 3 * σyy, "--", label="Reference", color="red", linewidth=2, fillstyle="none", markevery=25)
    ax[1].plot(xx, μyy - 3 * σyy, "--", label="Reference", color="red", linewidth=2, fillstyle="none", markevery=25)
    
    #plot the second picture
    x_w = zeros(length(objs[1].logx_w[1]), lines)
    x_mean = zeros(length(objs[1].x_mean[1]), lines)
    axx_cov, bxx_cov = size(objs[1].xx_cov[1])
    xx_cov = zeros(axx_cov, bxx_cov, lines)
    yy = zeros(length(yy_ref), lines)
    μyy = zeros(length(yy_ref))
    σyy = zeros(length(yy_ref))
    for iter = 0 : N_iter 
        error_mean = 0
        for i = 1 : lines
            x_w[:,i] = exp.(objs[i].logx_w[iter+1]); x_w[:,i] /= sum(x_w[:,i])
            x_mean[:,i] = objs[i].x_mean[iter+1]
            xx_cov[:,:,i] = objs[i].xx_cov[iter+1]
            yy[:,i] = Gaussian_mixture_1d(x_w[:,i], x_mean[:,i], xx_cov[:,:,i],  xx)
            error_mean += norm(yy[:,i] - yy_ref,1)*dx
        end
        error[2, iter+1] = error_mean / lines
        for l = 1 : length(yy_ref)
            μyy[l] = sum(yy[l,:]) / lines
            σyy[l] = norm(yy[l,:] - μyy[l] * ones(lines), 2) / sqrt(lines - 1)
        end
    end
    ax[2].plot(xx, μyy, label="Reference", color="red", linewidth=2, fillstyle="none", markevery=25)
    ax[2].plot(xx, μyy + 3 * σyy, "--", label="Reference", color="red", linewidth=2, fillstyle="none", markevery=25)
    ax[2].plot(xx, μyy - 3 * σyy, "--", label="Reference", color="red", linewidth=2, fillstyle="none", markevery=25)
    
    ax[N_obj+1].semilogy(Array(0:N_iter), error[1, :], label=objs[1].name*" (K="*string(size(half_objs[1].x_mean[1], 1))*")")
    ax[N_obj+1].semilogy(Array(0:N_iter), error[2, :], label=objs[2].name*" (K="*string(size(objs[1].x_mean[1], 1))*")")
    ax[N_obj+1].legend()
end


function Gaussian_density_2d(x_mean::Array{FT,1}, inv_sqrt_xx_cov, X, Y) where {FT<:AbstractFloat}
    dx = [X[:]' ; Y[:]'] - repeat(x_mean, 1, length(X))

    return reshape( exp.( -1/2*sum(dx .* ((inv_sqrt_xx_cov'*inv_sqrt_xx_cov)*dx), dims=1)) .* abs(det(inv_sqrt_xx_cov)), size(X))


end
function Gaussian_mixture_2d(x_w, x_mean, xx_cov,  X, Y)
    N_modes = length(x_w)
    inv_sqrt_xx_cov = [compute_sqrt_matrix(xx_cov[im,:,:]; type="Cholesky")[2] for im = 1:N_modes]
    # 2d Gaussian plot
    dx, dy = X[2,1] - X[1,1], Y[1,2] - Y[1,1]
    N_x, N_y = size(X)
    Z = zeros(N_x, N_y)
    
    
    for im = 1:N_modes
        Z .+= x_w[im]*Gaussian_density_2d(x_mean[im,:], inv_sqrt_xx_cov[im], X, Y)
    end

    Z = Z/(sum(Z)*dx*dy)
    
    return Z
    
end


function posterior_BIP_2d(func_F, X, Y)
    dx, dy = X[2,1] - X[1,1], Y[1,2] - Y[1,1]
    N_x, N_y = size(X)
    Z = zeros(N_x, N_y)
    for i = 1:N_x
        for j = 1:N_y
            F = func_F([X[i,j] ; Y[i,j]])
            Z[i,j] = exp(-F'*F/2)
        end
    end
    Z /= (sum(Z)*dx*dy) 

    return Z
end


function posterior_2d(func_V, X, Y)
    dx, dy = X[2,1] - X[1,1], Y[1,2] - Y[1,1]
    N_x, N_y = size(X)
    Z = zeros(N_x, N_y)
    for i = 1:N_x
        for j = 1:N_y
            V = func_V([X[i,j] ; Y[i,j]])
            Z[i,j] = exp(-V)
        end
    end
    Z /= (sum(Z)*dx*dy) 

    return Z
end



function visualization_2d(ax; Nx=2000, Ny=2000, x_lim=[-4.0,4.0], y_lim=[-4.0,4.0], func_F = nothing, func_V = nothing, objs=nothing)

    # visualization 
    x_min, x_max = x_lim
    y_min, y_max = y_lim

    xx = LinRange(x_min, x_max, Nx)
    yy = LinRange(y_min, y_max, Ny)
    dx, dy = xx[2] - xx[1], yy[2] - yy[1]
    X,Y = repeat(xx, 1, Ny), repeat(yy, 1, Nx)'   #'

    Z_ref = (func_V === nothing ? posterior_BIP_2d(func_F, X, Y) : posterior_2d(func_V, X, Y))
    color_lim = (minimum(Z_ref), maximum(Z_ref))
    ax[1].pcolormesh(X, Y, Z_ref, cmap="viridis", clim=color_lim)


   
    N_obj = length(objs)
    
    N_iter = length(objs[1].logx_w) - 1
    error = zeros(N_obj, N_iter+1)
        
    for (iobj, obj) in enumerate(objs)
        for iter = 0:N_iter  
            x_w = exp.(obj.logx_w[iter+1]); x_w /= sum(x_w)
            x_mean = obj.x_mean[iter+1]
            xx_cov = obj.xx_cov[iter+1]
            Z = Gaussian_mixture_2d(x_w, x_mean, xx_cov,  X, Y)
            error[iobj, iter+1] = norm(Z - Z_ref,1)*dx*dy
            
            if iter == N_iter
                ax[1+iobj].pcolormesh(X, Y, Z, cmap="viridis", clim=color_lim)
                N_modes = size(x_mean, 1)
                

                ax[1+iobj].scatter([obj.x_mean[1][:,1];], [obj.x_mean[1][:,2];], marker="x", color="grey") 
                ax[1+iobj].scatter([x_mean[:,1];], [x_mean[:,2];], marker="o", color="red", facecolors="none")

            end
        end
        
    end
    for i_obj = 1:N_obj
        ax[N_obj+2].semilogy(Array(0:N_iter), error[i_obj, :], label=objs[i_obj].name*" (K="*string(size(objs[i_obj].x_mean[1], 1))*")")
    end
    ax[N_obj+2].legend()
end



##########
function Gaussian_mixture(x, args)
    x_w, x_mean, inv_sqrt_x_cov = args
    # C = L L.T
    # C^-1 = L^-TL^-1
    N_x = size(x_mean, 2)
    ρ = 0
    for im = 1:length(x_w)
        ρ += x_w[im]*exp(-0.5*(x-x_mean[im,:])'*(inv_sqrt_x_cov[im]'*inv_sqrt_x_cov[im]*(x-x_mean[im,:])))/det(inv_sqrt_x_cov[im])
    end
    return log(ρ) - N_x/2*log(2*π)
end


function Gaussian_mixture_V(θ, args)
    return -Gaussian_mixture(θ, args), 
           -ForwardDiff.gradient(x -> Gaussian_mixture(x, args), θ), 
           -ForwardDiff.hessian(x -> Gaussian_mixture(x, args), θ)
end
##########


##########
function Gaussian_mixture(x, args)
    x_w, x_mean, inv_sqrt_x_cov = args
    # C = L L.T
    # C^-1 = L^-TL^-1
    N_x = size(x_mean, 2)
    ρ = 0
    for im = 1:length(x_w)
        ρ += x_w[im]*exp(-0.5*(x-x_mean[im,:])'*(inv_sqrt_x_cov[im]'*inv_sqrt_x_cov[im]*(x-x_mean[im,:])))/det(inv_sqrt_x_cov[im])
    end
    return log(ρ) - N_x/2*log(2*π)
end



function Gaussian_mixture_V(θ, args)
    return -Gaussian_mixture(θ, args), 
           -ForwardDiff.gradient(x -> Gaussian_mixture(x, args), θ), 
           -ForwardDiff.hessian(x -> Gaussian_mixture(x, args), θ)
end
##########


function G(θ, arg, Gtype = "Gaussian")
    K = ones(length(θ)-2,2)
    if Gtype == "Gaussian"
        A = arg
        return A*θ
    # elseif Gtype == "Double_modes"
    #     return [(θ[1]- θ[2])^2 ; (θ[1] + θ[2])^2]
    elseif Gtype == "Four_modes"
        return [(θ[1]- θ[2])^2 ; (θ[1] + θ[2])^2; θ[1:2]; θ[3:end]-K*θ[1:2]] 
    elseif Gtype == "Circle"
        A = arg
        return [θ'A*θ]
    elseif Gtype == "Banana"
        λ = arg
        return [λ*(θ[2] - θ[1]^2); θ[1]]
    elseif Gtype == "Double_banana"
        λ = arg
        return [log( λ*(θ[2] - θ[1]^2)^2 + (1 - θ[1])^2 ); θ[1]; θ[2]]
    else
        print("Error in function G")
    end
end


function F(θ, args)
    y, ση, arg, Gtype = args
    Gθ = G(θ, arg, Gtype )
    return (y - Gθ) ./ ση
end

function info_F(Gtype)
    if Gtype == "Gaussian"
        N_θ, N_f = 2, 2
    # elseif Gtype == "Double_modes"
    #     N_θ, N_f = 2, 2
    elseif Gtype == "Four_modes"
        N_θ, N_f = 2, 4
    elseif Gtype == "Circle"
        N_θ, N_f = 2, 1
    elseif Gtype == "Banana"
        N_θ, N_f = 2, 2
    elseif Gtype == "Double_banana"
        N_θ, N_f = 2, 3
    else
        print("Error in function G")
    end
    
    return N_θ, N_f
end


function logrho(θ, args)
    Fθ = F(θ, args)
    return -0.5*norm(Fθ)^2
end


function V(θ, args)
    return -logrho(θ, args), 
           -ForwardDiff.gradient(x -> logrho(x, args), θ), 
           -ForwardDiff.hessian(x -> logrho(x, args), θ)
end

##########
function Gaussian_mixture_VI(func_V, func_F, w0, μ0, Σ0; N_iter = 100, dt = 1.0e-3)

    N_modes, N_θ = size(μ0)
    

    
    T =  N_iter * dt
    N_modes = 1
    x0_w = w0
    x0_mean = μ0
    xx0_cov = Σ0
    sqrt_matrix_type = "Cholesky"
    quadrature_type_GM = "cubature_transform_o5"
    
    objs = []

    if func_V !== nothing
#         gmgdobj = GMGD_Run(
#         func_V, 
#         T,
#         N_iter,
#         # Initial condition
#         x0_w, x0_mean, xx0_cov;
#         sqrt_matrix_type = sqrt_matrix_type,
#         # setup for Gaussian mixture part
#         quadrature_type_GM = quadrature_type_GM,
#         # setup for potential function part
#         Bayesian_inverse_problem = false, 
#         quadrature_type = "cubature_transform_o3")
        
        
        gmgdobj = GMGD_Run(
        func_V, 
        T,
        N_iter,
        # Initial condition
        x0_w, x0_mean, xx0_cov;
        sqrt_matrix_type = sqrt_matrix_type,
        # setup for Gaussian mixture part
        quadrature_type_GM = "mean_point",
        # setup for potential function part
        Bayesian_inverse_problem = false, 
        quadrature_type = "mean_point")
        
        push!(objs, gmgdobj)

    end

    if func_F !== nothing
        N_f = length(func_F(ones(N_θ)))
        gmgdobj_BIP = GMGD_Run(
        func_F, 
        T,
        N_iter,
        # Initial condition
        x0_w, x0_mean, xx0_cov;
        sqrt_matrix_type = sqrt_matrix_type,
        # setup for Gaussian mixture part
        quadrature_type_GM = "mean_point",
        # setup for potential function part
        Bayesian_inverse_problem = true, 
        N_f = N_f,
        quadrature_type = "unscented_transform",
        c_weight_BIP = 1.0e-3,
        w_min=1e-10)
        
        push!(objs, gmgdobj_BIP)

    end

    return objs
end


N_modes_array = [10; 20; 40]
fig, ax = PyPlot.subplots(nrows=5, ncols=length(N_modes_array)+2, sharex=false, sharey=false, figsize=(20,16))

    
Random.seed!(11);
#Random.seed!(111);
N_modes = N_modes_array[end]
x0_w  = ones(N_modes)/N_modes
μ0, Σ0 = [0.0; 0.0], [1.0 0.0; 0.0 1.0]
N_x = length(μ0)
x0_mean, xx0_cov = zeros(N_modes, N_x), zeros(N_modes, N_x, N_x)
for im = 1:N_modes
    x0_mean[im, :]    .= rand(MvNormal(zeros(N_x), Σ0)) + μ0
    xx0_cov[im, :, :] .= Σ0
end




N_iter = 500
Nx, Ny = 100,100

# ση = 1.0
# Gtype = "Gaussian"
# A = [1.0 1.0; 1.0 2.0]
# y = [0.0; 1.0]
# func_args = (y, ση, A , Gtype)
# func_F(x) = F(x, func_args)
# func_dV(x) = V(x, func_args)
# #objs = (Gaussian_mixture_VI(func_dV, func_F, x0_w[1:div(N_modes,2)], x0_mean[1:div(N_modes,2),:], xx0_cov[1:div(N_modes,2),:,:]; N_iter = N_iter, dt = 1e-1)[1],
# #        Gaussian_mixture_VI(func_dV, func_F, x0_w, x0_mean, xx0_cov; N_iter = N_iter, dt = 1e-1)[1])
# objs = [Gaussian_mixture_VI(func_dV, func_F, x0_w[1:N_modes], x0_mean[1:N_modes,:], xx0_cov[1:N_modes,:,:]; N_iter = N_iter, dt = 1e-1)[1] 
#         for N_modes in N_modes_array]
# visualization_2d(ax[1,:]; Nx = Nx, Ny = Ny, x_lim=[-7.0, 5.0], y_lim=[-4.0, 5.0], func_F=func_F, objs=objs)



# # ση = 0.1
# # Gtype = "Double_modes"
# # y = [0.0; 1.0]
# # func_args = (y, ση, 0, Gtype)
# # func_F(x) = F(x, func_args)
# # func_dV(x) = V(x, func_args)
# # objs = (Gaussian_mixture_VI(func_dV, func_F, x0_w[1:div(N_modes,2)], x0_mean[1:div(N_modes,2),:], xx0_cov[1:div(N_modes,2),:,:]; N_iter = N_iter, dt = 1e-1)[1],
# #         Gaussian_mixture_VI(func_dV, func_F, x0_w, x0_mean, xx0_cov; N_iter = N_iter, dt = 1e-1)[1])
# # visualization_2d(ax[2,:]; Nx = Nx, Ny = Ny, x_lim=[-1.0, 1.0], y_lim=[-1.5, 1.5], func_F=func_F, objs=objs)


# ση = 1.0
# Gtype = "Four_modes"
# y = [4.2297; 4.2297; 0.5; 0.0; zeros(N_x-2)]
# func_args = (y, ση, 0, Gtype)
# func_F(x) = F(x, func_args)
# func_dV(x) = V(x, func_args)
# #objs = (Gaussian_mixture_VI(func_dV, func_F, x0_w[1:div(N_modes,2)], x0_mean[1:div(N_modes,2),:], xx0_cov[1:div(N_modes,2),:,:]; N_iter = N_iter, dt = 1e-1)[1],
# #        Gaussian_mixture_VI(func_dV, func_F, x0_w, x0_mean, xx0_cov; N_iter = N_iter, dt = 1e-1)[1])
# objs = [Gaussian_mixture_VI(func_dV, func_F, x0_w[1:N_modes], x0_mean[1:N_modes,:], xx0_cov[1:N_modes,:,:]; N_iter = N_iter, dt = 1e-1)[1] 
#         for N_modes in N_modes_array]
# #objs = [Gaussian_mixture_VI(nothing, func_F, x0_w[1:N_modes], x0_mean[1:N_modes,:], xx0_cov[1:N_modes,:,:]; N_iter = N_iter, dt = 1e-1)[1] 
# #        for N_modes in N_modes_array]
# visualization_2d(ax[2,:]; Nx = Nx, Ny = Ny, x_lim=[-4.0, 4.0], y_lim=[-4, 4], func_F=func_F, objs=objs)



# ση = 0.5
# Gtype = "Circle"
# A = [1.0 1.0; 1.0 2.0]
# y = [1.0;]
# func_args = (y, ση, A , Gtype)
# func_F(x) = F(x, func_args)
# func_dV(x) = V(x, func_args)
# μ0, Σ0 = [0.0; 0.0], [1.0 0.0; 0.0 1.0]
# # objs = (Gaussian_mixture_VI(func_dV, func_F, x0_w[1:div(N_modes,2)], x0_mean[1:div(N_modes,2),:], xx0_cov[1:div(N_modes,2),:,:]; N_iter = N_iter, dt = 1e-1)[1],
# #         Gaussian_mixture_VI(func_dV, func_F, x0_w, x0_mean, xx0_cov; N_iter = N_iter, dt = 1e-1)[1])
# objs = [Gaussian_mixture_VI(func_dV, func_F, x0_w[1:N_modes], x0_mean[1:N_modes,:], xx0_cov[1:N_modes,:,:]; N_iter = N_iter, dt = 1e-1)[1] 
#         for N_modes in N_modes_array]
# visualization_2d(ax[3,:]; Nx = Nx, Ny = Ny, x_lim=[-3.0, 3.0], y_lim=[-3.0, 3.0], func_F=func_F, objs=objs)



# ση = sqrt(10.0)
# Gtype = "Banana"
# λ = 10.0
# y = [0.0; 1.0]
# func_args = (y, ση, λ , Gtype)
# func_F(x) = F(x, func_args)
# func_dV(x) = V(x, func_args)
# μ0, Σ0 = [0.0; 0.0], [100.0 0.0; 0.0 100.0]
# # objs = (Gaussian_mixture_VI(func_dV, func_F, x0_w[1:div(N_modes,2)], x0_mean[1:div(N_modes,2),:], xx0_cov[1:div(N_modes,2),:,:]; N_iter = N_iter, dt = 1e-1)[1],
# #         Gaussian_mixture_VI(func_dV, func_F, x0_w, x0_mean, xx0_cov; N_iter = N_iter, dt = 1e-1)[1])
# objs = [Gaussian_mixture_VI(func_dV, func_F, x0_w[1:N_modes], x0_mean[1:N_modes,:], xx0_cov[1:N_modes,:,:]; N_iter = N_iter, dt = 1e-1)[1] 
#         for N_modes in N_modes_array]
# visualization_2d(ax[4,:]; Nx = Nx, Ny = Ny, x_lim=[-4.0, 4.0], y_lim=[-2.0, 10.0], func_F=func_F, objs=objs)



# ση = [0.3; 1.0; 1.0]
# Gtype = "Double_banana"
# λ = 100.0
# y = [log(λ+1); 0.0; 0.0]
# func_args = (y, ση, λ , Gtype)
# func_F(x) = F(x, func_args)
# func_dV(x) = V(x, func_args)
# # objs = (Gaussian_mixture_VI(func_dV, func_F, x0_w[1:div(N_modes,2)], x0_mean[1:div(N_modes,2),:], xx0_cov[1:div(N_modes,2),:,:]; N_iter = N_iter, dt = 1e-1)[1],
# #         Gaussian_mixture_VI(func_dV, func_F, x0_w, x0_mean, xx0_cov; N_iter = N_iter, dt = 1e-1)[1])
# objs = [Gaussian_mixture_VI(func_dV, func_F, x0_w[1:N_modes], x0_mean[1:N_modes,:], xx0_cov[1:N_modes,:,:]; N_iter = N_iter, dt = 1e-1)[1] 
#         for N_modes in N_modes_array]
# visualization_2d(ax[5,:]; Nx = Nx, Ny = Ny, x_lim=[-3.0, 3.0], y_lim=[-3.0, 3.0], func_F=func_F, objs=objs)

        ση = 1.0
        Gtype = "Gaussian"
        dt = 1e-1
        A = [1.0 1.0; 1.0 2.0]
        y = [0.0; 1.0; zeros(N_x-2)]
        func_args = (y, ση, A , Gtype)
        func_F(x) = F(x, func_args)
        func_dPhi(x) = V(x, func_args)
        objs = [Gaussian_mixture_VI(func_dPhi, nothing, x0_w[1:N_modes], x0_mean[1:N_modes,:], xx0_cov[1:N_modes,:,:]; N_iter = N_iter, dt = dt)[1] 
                for N_modes in N_modes_array]
        # compute marginal distribution
        y_2d = y[1:2]
        func_args = (y_2d, ση, A , Gtype)
        func_F(x) = F(x, func_args)
        visualization_2d(ax[1,:]; Nx = Nx, Ny = Ny, x_lim=[-7.0, 5.0], y_lim=[-4.0, 5.0], func_F=func_F, objs=objs)



        ση = 1.0
        dt = 2e-3
        Gtype = "Four_modes"
        y = [4.2297; 4.2297; 0.5; 0.0; zeros(N_x-2)]
        func_args = (y, ση, 0, Gtype)
        func_F(x) = F(x, func_args)
        func_dPhi(x) = V(x, func_args)
        objs = [Gaussian_mixture_VI(func_dPhi, nothing, x0_w[1:N_modes], x0_mean[1:N_modes,:], xx0_cov[1:N_modes,:,:]; N_iter = N_iter, dt = dt)[1] 
                for N_modes in N_modes_array]
        # compute marginal distribution
        y_2d = y[1:4]
        func_args = (y_2d, ση, A , Gtype)
        func_F(x) = F(x, func_args)
        visualization_2d(ax[2,:]; Nx = Nx, Ny = Ny, x_lim=[-4.0, 4.0], y_lim=[-4, 4], func_F=func_F, objs=objs)



        ση = [0.5; ones(N_x-2)]
        Gtype = "Circle"
        dt = 5e-3
        A = [1.0 1.0; 1.0 2.0]
        y = [1.0; zeros(N_x-2)]
        func_args = (y, ση, A , Gtype)
        func_F(x) = F(x, func_args)
        func_dPhi(x) = V(x, func_args)
        μ0, Σ0 = [0.0; 0.0], [1.0 0.0; 0.0 1.0]
        objs = [Gaussian_mixture_VI(func_dPhi, nothing, x0_w[1:N_modes], x0_mean[1:N_modes,:], xx0_cov[1:N_modes,:,:]; N_iter = N_iter, dt = dt)[1] 
                for N_modes in N_modes_array]
        # compute marginal distribution
        y_2d, ση_2d = y[1:1], ση[1:1]
        func_args = (y_2d, ση_2d, A , Gtype)
        func_F(x) = F(x, func_args)
        visualization_2d(ax[3,:]; Nx = Nx, Ny = Ny, x_lim=[-3.0, 3.0], y_lim=[-3.0, 3.0], func_F=func_F, objs=objs)



        ση = [sqrt(10.0); sqrt(10.0); ones(N_x-2)]
        Gtype = "Banana"
        dt = 2e-3
        λ = 10.0
        y = [0.0; 1.0; zeros(N_x-2)]
        func_args = (y, ση, λ , Gtype)
        func_F(x) = F(x, func_args)
        func_dPhi(x) = V(x, func_args)
        μ0, Σ0 = [0.0; 0.0], [1.0 0.0; 0.0 1.0]
        objs = [Gaussian_mixture_VI(func_dPhi, nothing, x0_w[1:N_modes], x0_mean[1:N_modes,:], xx0_cov[1:N_modes,:,:]; N_iter = N_iter, dt = dt)[1] 
                for N_modes in N_modes_array]
        # compute marginal distribution
        y_2d, ση_2d = y[1:2], ση[1:2]
        func_args = (y_2d, ση_2d, λ , Gtype)
        func_F(x) = F(x, func_args)
        visualization_2d(ax[4,:]; Nx = Nx, Ny = Ny, x_lim=[-4.0, 4.0], y_lim=[-2.0, 10.0], func_F=func_F, objs=objs)



        ση = [0.3; 1.0; 1.0; ones(N_x-2)]
        Gtype = "Double_banana"
        dt = 1e-5
        λ = 100.0
        y = [log(λ+1); 0.0; 0.0; zeros(N_x-2)]
        func_args = (y, ση, λ , Gtype)
        func_F(x) = F(x, func_args)
        func_dPhi(x) = V(x, func_args)
        objs = [Gaussian_mixture_VI(func_dPhi, nothing, x0_w[1:N_modes], x0_mean[1:N_modes,:], xx0_cov[1:N_modes,:,:]; N_iter = N_iter, dt = dt)[1] 
                for N_modes in N_modes_array]
        # compute marginal distribution
        y_2d, ση_2d = y[1:3], ση[1:3]
        func_args = (y_2d, ση_2d, λ , Gtype)
        func_F(x) = F(x, func_args)
        visualization_2d(ax[5,:]; Nx = Nx, Ny = Ny, x_lim=[-3.0, 3.0], y_lim=[-3.0, 3.0], func_F=func_F, objs=objs)

fig.tight_layout()
fig.savefig("DFGMGD.pdf")
