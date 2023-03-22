

using UltraNest 
using Distributions
using Random, LinearAlgebra, Statistics, StatsBase


println("Hello from ultranest2")


#generate some data

N =100
x = LinRange(400, 800, N)
yerr = 1.0
sigma = 500.0
data = zeros(N)
k = 1

for i in x

    local d = Normal(20*exp(-0.5*((i-500)/4.2)^2), sigma) #Construct the distribution type
    local y = rand(d, 1)[1]
    data[k] = y
    global k = k + 1

    #println(i, " ", y)

#println(k)

end 




function my_prior_transform(cube)
    params = copy(cube)

    # transform location parameter: uniform prior
    lo = 400
    hi = 800
    params[1] = cube[1] * (hi - lo) + lo

    # transform amplitude parameter: log-uniform prior
    lo = 0.1
    hi = 100
    params[2] = 10^(cube[2] * (log10(hi) - log10(lo)) + log10(lo))

    return params

end

function my_likelihood(params)
    #println("WELCOME TO MY LIKELIHOOD")
    #println(params)
    location, amplitude = params
    # compute intensity at every x position according to the model
    y_model = amplitude * exp.(-0.5 .* ((x .- location)./4.0).^2)

    # compare model and data with gaussian likelihood:
    like = -0.5 * sum((((y_model .- data)./yerr).^2))

    return like

end 



param_names = ["location", "amplitude"]

data = vcat(
    rand(Normal(-1.0, 0.5), 500),
    rand(Normal( 2.0, 0.5), 1000)
)



hist = append!(Histogram(-2:0.1:4), data)



function fit_function(p::NamedTuple{(:a, :mu, :sigma)}, x::Real)
    p.a[1] * pdf(Normal(p.mu[1], p.sigma), x) +
    p.a[2] * pdf(Normal(p.mu[2], p.sigma), x)
end




likelihood = let h = hist, f = fit_function
    # Histogram counts for each bin as an array:
    observed_counts = h.weights

    # Histogram binning:
    bin_edges = h.edges[1]
    bin_edges_left = bin_edges[1:end-1]
    bin_edges_right = bin_edges[2:end]
    bin_widths = bin_edges_right - bin_edges_left
    bin_centers = (bin_edges_right + bin_edges_left) / 2

    params -> begin
        #println(i)
        # Log-likelihood for a single bin:
        function bin_log_likelihood(i)
            # Simple mid-point rule integration of fit function `f` over bin:
            expected_counts = bin_widths[i] * f(params, bin_centers[i])
            logpdf(Poisson(expected_counts), observed_counts[i])
        end

        # Sum log-likelihood over bins:
        idxs = eachindex(observed_counts)
        ll_value = bin_log_likelihood(idxs[1])
        for i in idxs[2:end]
            ll_value += bin_log_likelihood(i)
        end

        # Wrap `ll_value` in `LogDVal` so BAT knows it's a log density-value.
        return LogDVal(ll_value)
    end
end



println(typeof(likelihood))


true_par_values = (a = [500, 1000], mu = (-1.0, 2.0), sigma = 0.5)

likelihood(true_par_values)


















# sampler = ultranest.ReactiveNestedSampler(param_names, my_likelihood, my_prior_transform)



# # sampler = ultranest.ReactiveNestedSampler(paramnames, mylikelihood, transform=mytransform, vectorized=true)
# results = sampler.run()
# print("result has these keys:", keys(results), "\n")

# println("Sampler results is")
# println(results["posterior"])

# sampler.print_results()

# println("NOW PLOT")
# plt = sampler.plot()
# display(plt)






