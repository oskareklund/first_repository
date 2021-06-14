# Example program "BFFG for continuous time observations"

using Statistics
using Random
using Colors
using Test 

## Simulating from system of SDEs and computing via Girsanov μ = E[f(X,Y) | Y=y] (whole path of Y is given)


T = 10.0
Δt = 0.01
t = 0.0:Δt:T
n = length(t)

# Define the model

"""
   b_1(x,θ)

Drift of the process `X`
"""
function b_1(x, θ) 
    return -θ[1]*x - θ[2]*sin(2pi*x)
end

function H(x,y)
    return x
end

function simulate_X_and_Y(x0, y0, n, Δt, θ)
    x = zeros(n)
    y = zeros(n)
    x[1] = x0
    y[1] = y0
    for i in 2:n
        x[i] = x[i-1] + b_1(x[i-1],θ)*Δt + sqrt(Δt)*randn()
        y[i] = y[i-1] + H(x[i-1],y[i-1])*Δt + sqrt(Δt)*randn()
    end
    return x, y
end

function simulate_X(x0, n,Δt,θ, Z = randn(n-1))
    x = zeros(n)
    x[1] = x0
    for i in 2:n
        x[i] = x[i-1] + b_1(x[i-1],θ)*Δt + sqrt(Δt)*Z[i-1]
    end
    return x
end


# Changes of measures

"""
    integral1(t,x,y)

Compute `∫ Y dX`
"""
function integral1(t,x,y)
    result = 0.0
    for i in 2:length(t)
        result = result + y[i-1]*(x[i] - x[i-1])
    end
    return result
end

"""
    integral2(t,x,y)

Compute `∫ X^2 ds`
"""
function integral2(t,x,y)
    result = 0.0
    for i in 2:length(t)
        result = result + x[i]^2*(t[i]-t[i-1])
    end
    return result
end

"""
    Ψ(t,x,y)
    Ψ(t,x,y,H)
    
Change of measure `exp([YX] - ∫ Y dX  - 0.5∫ X^2 ds)` between (X, Ỹ) to (X, Y), see overleaf.
"""
function Ψ(t,x,y)
    n = length(t)
    return exp(y[n]*x[n] - y[1]*x[1] - integral1(t,x,y) - 1/2*integral2(t,x,y))
end

function Ψ(t, x, y, H)
    result = H(x[end],y[end])*y[end] - H(x[1],y[1])*y[1] # times varsigma.....
    for i in 2:length(t) # stoch
        result += (-1)*y[i-1]*(H(x[i], y[i]) - H(x[i-1], y[i-1]))
        result += (-1/2) * H(x[i-1], y[i-1])^2 * (t[i] - t[i-1])
    end
    return exp(result)
end

"""
Change of measure between `Xᵒ` and `X`
"""
function Φ(t, x, ν, P, Q, Z)
    result = 0.0
    Qroot = sqrt(Q)
    for i in 2:length(t) # stoch
        
        result += Qroot*inv(P[i-1])*(ν[i-1] - x[i-1])*sqrt((t[i]-t[i-1]))*Z[i-1]
        result += (-1/2)*(ν[i-1] - x[i-1])'*inv(P[i-1])'*Q*inv(P[i-1])*(ν[i-1] - x[i-1])*(t[i]-t[i-1])
    end
    return exp(result)
end



# Backward filter

"""
    solve_ν_and_P(n, H, B, β, y, Q, νT, PT) 

for 

    dX[t] = (BX[t] + β)*dt + σ dWt
    dY[t] = H*X[t]dt + Wt # R = I in the paper

where `Q = σσ'`
"""
function solve_ν_and_P(n, H, B, β, y, Q, νT, PT)
    ν = zeros(n)
    P = zeros(n)
    ν[end] = νT
    P[end] = PT
    for i in n-1:-1:1    
        P[i] = P[i+1] - (B*P[i+1] + P[i+1]*B' - Q + P[i+1]*H'*H*P[i+1])*Δt 
        ν[i] = ν[i+1] - (B*ν[i+1] + β - P[i+1]*H'*((y[i+1]-y[i])/Δt - H*ν[i+1]))*Δt
    end
    return ν, P
end

# Proposal

function simulate_Xᵒ(x0, n, Δt, θ, σ, ν, P, Z = randn(n-1))
    x = zeros(n)
    x[1] = x0
    for i in 2:n
        pull = σ*σ'*P[i-1]\(ν[i-1] - x[i-1])
        x[i] = x[i-1] + (b_1(x[i-1], θ) + pull)*Δt + σ*sqrt(Δt)*Z[i-1]
    end
    return x
end




####################################################################################

# Generate observations

θ = [0.5, 2.]
x0 = 1.0
y0 = 0.0
Random.seed!(4)

x_true, y_true = simulate_X_and_Y(x0, y0, n, Δt, θ)

y_true = y_true + 4*t

using Plots

fig1 = plot(t, x_true, linewidth=2.0, color=:blue)
for i in 1:100
    Z = randn(n-1)
    x = simulate_X(x0, length(t), Δt, θ, Z) 
    plot!(fig1, t, x, color=RGBA(1,0,0,0.1))
end



# Crank Nicolson
num_xs = 10000
xs = zeros(num_xs, length(t))
phis = zeros(num_xs)

Z = randn(n-1)
xs[1,:] = x = simulate_X(x0, length(t), Δt, θ, Z) 
phis[1] = Ψ(t, x, y_true, H)

# @test Ψ(t, x, y_true, H) ≈ Ψ(t, x, y_true)  atol=1e-2

ρ = 0.8

acc_count = 0

# Get sample from X | y_true

for i in 2:num_xs
    global Z
    global acc_count
    Z_new = ρ*Z + sqrt(1-ρ^2)*randn(n-1) # Symmetric proposal
    x = simulate_X(x0, length(t), Δt, θ, Z_new)  
    phi = Ψ(t, x, y_true, H)
    if rand() < phi/phis[i-1] # Metropolis-Hastings
        xs[i,:] = x 
        phis[i] = phi
        Z = Z_new
        acc_count += 1
    else
        xs[i,:] = xs[i-1,:]
        phis[i] = phis[i-1]
    end
end

println("Acceptance probability: ", acc_count/num_xs)
fig2 = plot(t, y_true, color=:blue, legend = false, show = true)

for i in 1:(max(1,num_xs÷100)):num_xs 
    plot!(t, xs[i,:], color = RGBA(1,0,0,0.1), legend = false)
end
display(fig2)
#savefig(fig2,"fig2test.png")


##
#a=fail
##


x_mean = vec(mean(xs, dims = 1))
x_std = vec(std(xs, dims = 1))




Hop = 1.0
B = -θ[1]
β = 0.0
Q = 1.0
νT = 0.0
PT = 1*(1-exp(2B*t[end]))
ν, P = solve_ν_and_P(n, Hop, B, β, y_true, Q, νT, PT)
x2 = simulate_Xᵒ(x0, n, Δt, θ, 1.0 #=σ=#, ν, P,  randn(n-1))
x_mean2 = simulate_Xᵒ(x0, n, Δt, θ, 1.0 #=σ=#, ν, P, zeros(n-1))


######################################################################## 20/5, 21/5
# Next step: Picture of E[f(X)|Y=y] = E[f(Xcirc)*Psi/C*Phi] Done here.

# Phi, Psis, Xcirc created.

num_xs = 10000
phipsis = zeros(num_xs) # This is the new "phis" 
xcircs = zeros(num_xs,length(t))
ys = zeros(num_xs,length(t))

Hop = 1.0
B = -θ[1]
β = 0.0
Q = 1.0
νT = 0.0
PT = 20*(1-exp(2B*t[end]))

ν, P = solve_ν_and_P(n, Hop, B, β, y_true, Q, νT, PT) # To have initial v and P
# filter wrong backwards
νalter, Palter = solve_ν_and_P(n, Hop, 0.5*B, β, y_true, Q, νT, PT) # To have initial v and P

# Was fixing errors above 9/6 13:57, correct until here.

Z = randn(n-1)
xcircs[1,:] = xcirc = simulate_Xᵒ(x0,n,Δt,θ,1.0 #=σ=#,νalter,Palter)
phipsi = Ψ(t,xcirc,y_true,H)/Φ(t,xcirc,νalter,Palter,Q,Z)
phipsi *= exp(-(νT - xcirc[end])'*inv(PT)*(νT - xcirc[end])/2)
phipsis[1] = phipsi

ρ = 0.8

acc_count = 0

L = [0.0]
for i in 2:num_xs
    global Z
    global acc_count
    Z_new = ρ*Z + sqrt(1-ρ^2)*randn(n-1)
    xcirc = simulate_Xᵒ(x0,n,Δt,θ,1.0 #=σ=#,νalter,Palter, Z_new)
    phipsi = Ψ(t,xcirc,y_true,H)/Φ(t,xcirc,νalter,Palter,Q,Z)
    phipsi *= exp(-(νT - xcirc[end])'*inv(PT)*(νT - xcirc[end])/2)
    push!(L, phipsi/phipsis[i-1])
    if rand() < phipsi/phipsis[i-1]
        xcircs[i,:] = xcirc
        phipsis[i] = phipsi
        Z = Z_new
        acc_count += 1
    else
        xcircs[i,:] = xcircs[i-1,:]
        phipsis[i] = phipsis[i-1]
    end
end

println("Acceptance probability: ", acc_count/num_xs)
p = plot(t, y_true, color=:blue, legend = false, show = true)

for i in 1:(max(1,num_xs÷100)):num_xs 
    plot!(t,xcircs[i,:], color = RGBA(1,0,0,0.1), legend = false)
end
#savefig(p,"fig3.png")

x_mean = vec(mean(xcircs, dims = 1)) # The true mean of the conditional process
x_std = vec(std(xcircs, dims = 1))
 
#plot!(t,x_mean, color=:green)
#plot!(t,x_mean + 1.96*x_std,color=:black)
#plot!(t,x_mean - 1.96*x_std,color=:black)
 
