using ForwardDiff
using KernelFunctions
using LinearAlgebra

"""
Xt: 'd×J' matrix which save J paticals(each in R^d) at current time.
k: kernel function
logRatio: function R^d->R ,where Ratio is \
N: how many iteration (N=1/Δt)
output::String "Last" for returning the last paticals, "History" for returning the historical paticals.
"""
function generateDataKernel(
    Xt::Matrix,
    kernel::Kernel=RationalQuadraticKernel(α=1/2),
    )

    J=size(Xt,2)
    ∇k1(x,y)=ForwardDiff.gradient(u->kernel(u,y),x)
    Kt(x)=[kernel(x,Xt[:,j]) for j=1:J]
    ∇Kt(x)=vcat([(∇k1(x,Xt[:,j]))' for j=1:J]...)
    return Kt,∇Kt
end
"""
Xt: 'd×J' matrix which save J paticals(each in R^d) at current time.
kernel: kernel function
logRatio: function R^d->R ,where Ratio is
N: how many iteration (N=1/Δt)
output::String  :"Last" for returning the last paticals, "History" for returning the historical paticals.
"""
function FKRFlowEuler(
    X0::Matrix,
    logRatio::Function;
    N::Int=100,
    kernel::Kernel=RationalQuadraticKernel(α=1/2),
    output::String="Last"
    )
    
    Xt=copy(X0)
    Xhist=copy(X0)
    Δt=1/N
    J=size(Xt,2)
    eps=1e-4
    
    for iter=1:N
        Kt,∇Kt=generateDataKernel(Xt,kernel)
        Mt=sum(∇Kt(Xt[:,j])*∇Kt(Xt[:,j])' for j=1:J)/J
        Mt=Mt+eps*I
        aver_logRatio=sum(logRatio(Xt[:,i]) for i=1:J)/J
        aver_vec=sum( (logRatio(Xt[:,j])-aver_logRatio)*Kt(Xt[:,j]) for j=1:J)/J
        ΔX=zeros(size(X0))
        for i=1:J
            ΔX[:,i]=Δt*(∇Kt(Xt[:,i])')*(Mt\aver_vec)
        end
        Xt=Xt+ΔX
        if output=="History" 
            Xhist=cat(Xhist, Xt, dims=3)
        end
    end
    
    if output=="Last"
            return Xt
    else
            return Xhist 
    end
    
end