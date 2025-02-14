using LinearAlgebra
include("../Inversion/GaussianMixture.jl")

function Low_Rank_Gaussian_density(x, sqrt_xx_cov, cov_eps; D_inv = nothing) 
    """
        Efficiently calculate: Gaussian density exp(-x'Cx/2)/sqrt(det C),  
        where C = (sqrt_xx_cov)(sqrt_xx_cov)'+ eps*I 
        Notice we tolerate constant multiple error independent of x,C.
    """
    N_x, N_r = size(sqrt_xx_cov)
    if N_x <= N_r 
        C = sqrt_xx_cov*sqrt_xx_cov'+cov_eps*I # size=(N_x, N_x)
        z = C\x # Cz=x
        return exp(-dot(z,x)/2)/sqrt(det(C))
    else 
        # N_r < N_x   
        y = sqrt_xx_cov'*x
        if D_inv == nothing 
            D = sqrt_xx_cov'*sqrt_xx_cov + cov_eps*I   # size=(N_r, N_r)
            if !(det(D)>0)
                @error D,det(D)
                @show x, sqrt_xx_cov
            end
            temp1 = (-norm(x)^2 + dot(y,D\y))/(2*cov_eps)-log(cov_eps)*(N_x-N_r)/2
            temp2 = sqrt(det(D))
            return exp(temp1)/temp2
        else
            temp1 = (-norm(x)^2 + dot(y, D_inv*y))/(2*cov_eps)-log(cov_eps)*(N_x-N_r)/2
            temp2 = sqrt(det(D_inv))
            return exp(temp1)*temp2
        end
    end
end

function rank_scalar_approximation(eps0, Q0, R0, D0, N_r; rank_plus::Int = 3, A = nothing, eps_min = 0.01, method = "RandSVD")
    """Approximate A = eps0*I + Q0*Q0' + R0*D0*R0'  with  A = eps*I +QQ',
    based on PPCA method and eigenvalue decomposition(by RandSVD / Nystrom). """
    if method == "RandSVD"
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
        # D = D[1:N_r]
        eps = (trA-sum(D))/(N_x-N_r)
        eps = max(eps, eps_min)
        newQ = hcat( [sqrt(max(D[i]-eps,1.0e-6))*U[:,i] for i=1:N_r]... )
    elseif method == "Nystrom"
        if A == nothing
            N_x = size(Q0,1)
            Omega = randn(N_x, N_r+rank_plus)
            Y = eps0*Omega + Q0*(Q0'*Omega) + (R0.*D0')*(R0'*Omega)
            trA = eps0*N_x + sum(Q0.*Q0) + sum(R0.*(R0.*D0'))
        else
            trA = tr(A)
            N_x = size(A,1)
            Omega = randn(N_x, N_r+rank_plus)
            Y = A*Omega
        end 
        mu = sqrt(N_x)*norm(Y)*1.0e-6
        Y_mu = Y + mu*Omega
        C = cholesky(Hermitian(Omega'*Y_mu)).L
        B = (C\(Y_mu'))'
        U, D_sqrt, _ = svd(B)
        D = [max(D_sqrt[i]^2-mu,0)  for i=1:N_r]
        eps = (trA-sum(D))/(N_x-N_r)
        eps = max(eps, eps_min)
        newQ = hcat( [sqrt(max(D[i]-eps,0.01))*U[:,i] for i=1:N_r]... )
    else
        @error "Undefined rank_scalar_approximation method"
    end
    return eps, newQ
end

function new_rank_scalar_approximation(eps0, Q0, R0, D0, N_r; rank_plus::Int = 3, A = nothing, eps_min = 0.01, method = "RandSVD")
    """Approximate A = eps0*I + Q0*Q0' + R0*D0*R0'  with  A = eps*I +QQ',
    based on PPCA method and eigenvalue decomposition(by RandSVD / Nystrom). """
    if method == "RandSVD"
        if A == nothing
            N_x = size(Q0,1)
            Omega = randn(N_x, N_r+rank_plus)
            Y = eps0*Omega + Q0*(Q0'*Omega) + (R0.*D0')*(R0'*Omega)
            Yqr = qr(Y)
            Q = Matrix(Yqr.Q)  
            B = eps0*Q' + Q'*Q0*Q0' + Q'*(R0.*D0')*R0'
            trA = eps0*N_x + sum(Q0.*Q0) + sum(R0.*(R0.*D0'))
        else
            trA_inv = tr(inv(A))
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
        eps = (N_x-N_r)/(trA_inv-sum(1.0./D))
        eps = max(eps, eps_min)
        newQ = hcat( [sqrt(max(D[i]-eps,1.0e-6))*U[:,i] for i=1:N_r]... )
    else
        @error "Undefined rank_scalar_approximation method"
    end
    return eps, newQ
end

function Gaussian_mixture_sampler(logx_w, x_mean, xx_sqrt_cov, cov_eps, N_sample)
    x_w = exp.(logx_w)
    x_w /= sum(x_w)
    modes_dist = Categorical(x_w)
    modes_sample = rand(modes_dist,N_sample)
    N_modes, N_x, N_r = size(xx_sqrt_cov)
    xs = zeros(N_x,N_sample)
    for i = 1:N_sample
        im = modes_sample[i]
        xs[:,i] = xx_sqrt_cov[im,:,:]*randn(N_r)+sqrt(cov_eps[im])*randn(N_x)+x_mean[im,:]
    end
    return xs
end

function PGM_visualization_2d(ax; Nx=200, Ny=200, x_lim=[-4.0,4.0], y_lim=[-4.0,4.0], func_Phi = nothing, objs=nothing, label=nothing)

    # visualization 
    x_min, x_max = x_lim
    y_min, y_max = y_lim

    xx = LinRange(x_min, x_max, Nx)
    yy = LinRange(y_min, y_max, Ny)
    dx, dy = xx[2] - xx[1], yy[2] - yy[1]
    X,Y = repeat(xx, 1, Ny), repeat(yy, 1, Nx)'   #'

    Z_ref = posterior_2d(func_Phi, X, Y, "func_Phi")
    color_lim = (minimum(Z_ref), maximum(Z_ref))
    ax[1].pcolormesh(X, Y, Z_ref, cmap="viridis", clim=color_lim)

    N_obj = length(objs)
    N_iter = length(objs[1].logx_w) - 1
    error = zeros(N_obj, N_iter+1)
        
    for (iobj, obj) in enumerate(objs)
        N_modes = obj.N_modes
        for iter = 0:N_iter  
            x_w = exp.(obj.logx_w[iter+1]); x_w /= sum(x_w)
            x_mean = obj.x_mean[iter+1][:,1:2]
            xx_sqrt_cov = obj.xx_sqrt_cov[iter+1][:,1:2,:]
            cov_eps = obj.cov_eps[iter+1]
            xx_cov = zeros(N_modes,2,2)
            for im = 1:N_modes
                xx_cov[im,:,:] = xx_sqrt_cov[im,:,:]*xx_sqrt_cov[im,:,:]'+cov_eps[im]*I
            end
            Z = Gaussian_mixture_2d(x_w, x_mean, xx_cov,  X, Y)
            error[iobj, iter+1] = norm(Z - Z_ref,1)*dx*dy
            
            if iter == N_iter
                    
                ax[1+iobj].pcolormesh(X, Y, Z, cmap="viridis", clim=color_lim)
                N_modes = size(x_mean, 1)
                ax[1+iobj].scatter([obj.x_mean[1][:,1];], [obj.x_mean[1][:,2];], marker="x", color="grey", alpha=0.5) 
                ax[1+iobj].scatter([x_mean[:,1];], [x_mean[:,2];], marker="o", color="red", facecolors="none", alpha=0.5)
               
            end
        end
        
    end
    for i_obj = 1:N_obj
        ax[N_obj+2].semilogy(Array(0:N_iter), error[i_obj, :], 
                        label=(label===nothing ? label : label*" (K="*string(size(objs[i_obj].x_mean[1], 1))*")" ))   
   end
    # Get the current y-axis limits
    ymin, ymax = ax[N_obj+2].get_ylim()
    # Ensure the lower bound of y-ticks is below 0.1
    if ymin > 0.1
        ax[N_obj+2].set_ylim(0.1, ymax)  # Set the lower limit to a value below 0.1
    end
    if label!==nothing 
       ax[N_obj+2].legend()
    end
   
end