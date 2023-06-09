struct simdata
    xs
    x_baseline
    tvals
    group1
    group2
end

function generate_xs(n, p, true_u0, sol_group1, sol_group2; 
    t_start=1.5f0, t_end=10f0, maxntps = 10, dt=0.1f0, σ_var=0.1f0, σ_ind=0.5f0)

    # (1) generate artifical group labels
    groups = zeros(n)
    groups[randperm(n)[1:Int(floor(n/2))]] .= 1 # for groups of the same size
    group1 = findall(x -> x==1, groups)
    group2 = findall(x -> x==0, groups)

    # (2) generate artificial time stamps
    ntps = rand(1:maxntps-1, n)
    tvals = [sort(rand(t_start:dt:t_end,ntps[i])) for i in 1:n] # without 0 

    # (3) obtain true values as solutions of the ODEs at the initial time point and the drawn second time point 
    # check for equal number of variables:
    if p%2 != 0
        error("Please select an even number of variables")
    end
    # true starting point    
    z_t0_p1 = true_u0[1] # for variables 1-(p/2)
    z_t0_p2 = true_u0[2] # for variables (p/2+1)-p
    z_t0 = repeat([z_t0_p1, z_t0_p2], inner=Int(p/2))

    # now use ODE solutions to obtain true temporal development value
    # for all individuals in both variables u1 and u2
    z_later_ts = collect((i ∈ group1) ? (sol_group1[:,Int.(round.(tvals[i].*(1 ./dt)).+1)]) : (sol_group2[:,Int.(round.(tvals[i].*(1 ./dt)).+1)]) for i in 1:n)

    # (4) sample variable- specific and individual-specific errors at both time points
    # variable specific random effect (general difficulty measuring that specific variable)
    us = rand(Normal(0.0f0 ,σ_var), p) 

    xs = []
    for i in 1:n 
        # make time series structure, should have shape (p x ntps[i])
        cur_timeseries = zeros(Float32, (p, ntps[i]+1))
        for j in 1:p 
            cur_timeseries[j,1] = z_t0[j] + us[j] + randn(Float32) .* σ_ind
            for tp in 1:ntps[i]
                if j <= Int(p/2)
                    cur_timeseries[j,tp+1] = z_later_ts[i][1,tp] + us[j] + randn(Float32) .* σ_ind
                else
                    cur_timeseries[j,tp+1] = z_later_ts[i][2,tp] + us[j] + randn(Float32) .* σ_ind
                end
            end
        end
        push!(xs, cur_timeseries)
    end

    # (5) append 0 to tvals 
    tvals = map(x -> insert!(x, 1, 0.0f0), tvals)

    return xs, tvals, group1, group2
end

function generate_baseline(n, q, q_info, group1; σ_info=1.0f0, σ_noise=1.0f0)
    zs = fill(1.0f0,(n,1))
    zs[group1].= -1.0f0
    means = fill(0.0f0,n,q)
    means[:,1:q_info] .= zs

    vars=fill(σ_noise,q)
    vars[1:q_info] .= σ_info
    x_params = [cat([rand(Normal(means[i,j],vars[j])) for j in 1:q]..., dims=1) for i in 1:n]
    return x_params
end

function generate_baseline(n, q, q_info, group1, trueparams_group1, trueparams_group2; 
    σ_info=0.1f0, σ_noise=0.1f0)
    signs = fill(1,(n,1))
    signs[group1] .= -1
    z1s = zeros(Float32, (n,1))
    z1s[signs .== -1] .= trueparams_group1[1]
    z1s[signs .== 1] .= trueparams_group2[1]
    z2s = zeros(Float32, (n,1))
    z2s[signs .== -1] .= trueparams_group1[2]
    z2s[signs .== 1] .= trueparams_group2[2]

    means = zeros(Float32, (n,q))
    means[:,1:Int(floor(q_info/2))] .= z1s
    means[:,Int(floor(q_info/2))+1:q_info] .=z2s

    vars=fill(σ_noise, q)
    vars[1:q_info] .= σ_info
    x_params = [cat([rand(Normal(means[i,j],vars[j])) for j in 1:q]..., dims=1) for i in 1:n]
    return x_params
end