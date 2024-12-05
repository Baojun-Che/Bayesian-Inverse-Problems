using ForwardDiff
using KernelFunctions
using LinearAlgebra
using Statistics

#= Experiments show that RBF kernel performs significantly better that other kernel functions, 
and the bandwidth should be choosen to be med^2/(2*log(N+1)).
However, the bandwidth of RBF kernel in package "KernelFunctions" cannot be change()
Therefore, we manually write RBF kernel and its gradient with respect to the first variable.
=#
function α_adapt(data::Matrix)
    dim,N=size(data)
    dis=zeros(N,N)
    for i = 1:N, j = 1:N
        dis[i,j]=norm(data[:,i]-data[:,j])
    end
    return (median(dis)^2)/log(N+1)/2
end

function RBF_kernel(x,y;α::Real=1.0)
    return exp(-(norm(x-y)^2)/α) 
end 
function RBF_kernel_gradient_1(x,y;α::Real=1.0) # return \frac{\partial k(x,y)}{\partial x}.
    return -2*(x-y)/α*exp(-(norm(x-y)^2)/α)
end 

#=
X0: 'd×N' matrix which save N initial paticals(each in R^d)
∇log_prob: function R->R^d ,where prob is the density to be estimate
s0: initial step length
output::String "Last" for returning the last paticals, "History" for returning the historical paticals.
=#

function SVGD(
    X0::Matrix,
    ∇log_prob::Function;
    N_iter::Int=1000,
    s0::Real=0.1,
    output::String="Last",
    use_RMSProp::Bool=true
    )

    α=α_adapt(X0)
    k(x,y)=RBF_kernel(x,y,α=α)
    ∇k1(x,y)=RBF_kernel_gradient_1(x,y,α=α)

    N_ens=size(X0,2) # number of paticals
    X=copy(X0)
    
    #RMSProp
    G=zeros(size(X0)) # record gradient accumulation
    ϵ=1e-6 # avoid the phenomenon of zero division
    β=0.95  # decay rate of G

    Xhist=copy(X0)
    
    for iter=1:N_iter

        if iter%max(1,div(N_iter, 10)) == 0  @info "iter = ", iter, " / ", N_iter  end

        𝛟(x) = sum([k(X[:,j],x)*∇log_prob(X[:,j])+∇k1(X[:,j],x) for j=1:N_ens])/N_ens
        ΔX = hcat(𝛟.(eachcol(X))...)

        if use_RMSProp==true
            X += (iter==1 ?  s0*ΔX : s0*ΔX./sqrt.(G.+ϵ))
            G = G+(1-β)*ΔX.*ΔX
        else
            X += s0*ΔX
        end

        if output=="History" 
            Xhist=cat(Xhist, X, dims=3)
        end
    end

    if output=="Last"
        return X
    else
        return Xhist 
    end
end



function DF_SVGD(
    X0::Matrix,
    log_prob::Function,
    log_rho::Function;# log of surrogate function
    N_iter::Int=1000,
    s0::Real=0.1,
    output::String="Last",
    use_RMSProp::Bool=true
    )

    α=α_adapt(X0)
    
    ∇log_rho(x)=ForwardDiff.gradient(log_rho,x)
    N_ens=size(X0,2) # number of paticals
    X=copy(X0)
    
    #RMSProp
    G = zeros(size(X0)) # record gradient accumulation
    ϵ = 1e-6 # avoid the phenomenon of zero division
    β = 0.9  # decay rate of G
    
    Xhist=copy(X0)

    for iter=1:N_iter

        if iter%max(1,div(N_iter, 10)) == 0  
            @info "iter = ", iter, " / ", N_iter  
            α=α_adapt(X)
        end

        w(x) = exp(log_rho(x)-log_prob(x))
        Phi(x) = sum( w(X[:,j])*(RBF_kernel(X[:,j],x,α=α)*∇log_rho(X[:,j])+RBF_kernel_gradient_1(X[:,j],x,α=α)) for j=1:N_ens )
        ΔX = hcat(Phi.(eachcol(X))...)
        ΔX ./= sum(w(X[:,j]) for j=1:N_ens)

        if use_RMSProp==true
            X += (iter==1 ?  s0*ΔX : s0*ΔX./sqrt.(G.+ϵ))
            G = β*G+(1-β)*ΔX.*ΔX
        else
            X += s0*ΔX
        end

        if output=="History" 
            Xhist=cat(Xhist, X, dims=3)
        end
    end

    if output=="Last"
        return X
    else
        return Xhist 
    end
end


function ADF_SVGD( 
    X0::Matrix,
    log_prob::Function;
    N_iter::Int=1000,
    s0::Real=0.1,
    output::String="Last",
    use_RMSProp::Bool=false #Experiments show that RMSProp is not helpful in ADF-SVGD
    )

    N_x,N_ens=size(X0) # N_x:dimension, N_ens: number of paticals
    X=copy(X0)
    
    log_p0(x)=1

    #RMSProp
    G = zeros(size(X0)) # record gradient accumulation
    ϵ = 1e-6  # avoid the phenomenon of zero division
    β = 0.90 # decay rate of G
    
    Xhist=copy(X0)
    α=α_adapt(X)
    @show α

    for iter=1:N_iter

        temperature = min(2*(iter/N_iter),1)

        if iter%max(1,div(N_iter, 10)) == 0  
            @info "iter = ", iter, " / ", N_iter 
        end

        # Experiments show that its benificial to change α once at the 25%-50% of iterations. 
        if iter == div(N_iter, 3)  
            α=α_adapt(X) 
            @show α
        end

        #To reduce computation cost, we expanded the equation in ADF-SVGD algorithm.

        prob_ratio = zeros(N_ens,N_ens) #record the ratio p^t(x_m)/p^t(x_j) in [m,j] element
        kernel_values = zeros(N_ens,N_ens) #record the value of k(x_m,x_j) in [m,j] element
        kernel_gradients = zeros(N_x,N_ens,N_ens) #record the value of  ∇k1(x_m,x_j) in [m,j] element
        for m = 1:N_ens, j = 1:N_ens
            x_m = X[:,m]
            x_j = X[:,j]
            prob_ratio[m,j] = exp(temperature*(log_prob(x_m)-log_prob(x_j))+(1-temperature)*(log_p0(x_m)-log_p0(x_j)))
            kernel_values[m,j] = RBF_kernel(x_m,x_j,α=α)
            kernel_gradients[:,m,j] =  RBF_kernel_gradient_1(x_m,x_j,α=α)
        end
        
        H1 = zeros(N_x,N_ens)
        H2 = zeros(N_ens)
        for j = 1:N_ens
            H1[:,j] = sum(prob_ratio[m,j]*kernel_gradients[:,j,m] for m=1:N_ens)
            H2[j] = sum(prob_ratio[m,j]*kernel_values[m,j] for m=1:N_ens)
        end

        ΔX = zeros(size(X0))
        for i = 1:N_ens
            for j = 1:N_ens
                ΔX[:,i] += kernel_values[i,j]*H1[:,j] + H2[j]*kernel_gradients[:,j,i]
            end
        end
        Z = sum(kernel_values.*prob_ratio)
        ΔX /= Z 
        
        if use_RMSProp==true
            X += (iter==1 ?  s0*ΔX : s0*ΔX./sqrt.(G.+ϵ))
            G = β*G+(1-β)*ΔX.*ΔX
        else
            X += s0*ΔX
        end

        if output=="History" 
            Xhist=cat(Xhist, X, dims=3)
        end
    end

    if output=="Last"
        return X
    else
        return Xhist 
    end
end
