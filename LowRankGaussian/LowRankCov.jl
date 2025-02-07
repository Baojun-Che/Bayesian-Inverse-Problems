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
            temp1 = (-norm(x)^2 + dot(y,D\y))/(2*cov_eps)
            temp2 = sqrt(det(D))*cov_eps^((N_x-N_r)/2)
            return exp(temp1)/temp2
        else
            temp1 = (-norm(x)^2 + dot(y, D_inv*y))/(2*cov_eps)
            temp2 = sqrt(det(D_inv))
            temp3 = cov_eps^((N_x-N_r)/2)
            return exp(temp1)*temp2/temp3
        end
    end
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
                xx_cov[im,:,:]=xx_sqrt_cov[im,:,:]*xx_sqrt_cov[im,:,:]'+cov_eps[im]*I
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