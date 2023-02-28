#using Pkg
#Pkg.activate(".")
#Pkg.instantiate()
using DifferentialEquations  # For generating synthetic training data
using DynamicalSystems       # For calculating Lyapunov exponents (a great package with much more to it than this)
using ReservoirComputing     # Julia implementation of reservoir computers
using Printf                 # Nice C-style string formatting
using Plots
using CSV
using DataFrames
using NaNStatistics
using Statistics
using Hyperopt


a = 0.1
Wᵢₙ = WeightedLayer(scaling = a)

reservoir_size = 1000           # Number of nodes in the reservoir
spectral_radius = 1.0           # Good starting value
sparsity = 10 / reservoir_size  # Each node is connected to 10 others, on average

W = RandSparseReservoir(reservoir_size, radius = spectral_radius, sparsity = sparsity)


"""
    generate_esn(input_signal, reservoir_size = 1000, spectral_radius = 1.0, sparsity = 0.1, input_scale = 0.1)

Generate an Echo State Network consisting of the reservoir weights W and the input weights Wᵢₙ.
"""
function generate_esn(
    input_signal,
    reservoir_size = 1000,
    spectral_radius = 1.0,
    sparsity = 0.1,
    input_scale = 0.1,
)
    Wᵢₙ = WeightedLayer(scaling = input_scale)
    W = RandSparseReservoir(reservoir_size, radius = spectral_radius, sparsity = sparsity)
    return ESN(input_signal, reservoir = W, input_layer = Wᵢₙ)
end

ridge_param = 0.0  # OLS
training_method = StandardRidge(ridge_param)


"""
    train_esn!(esn, y, ridge_param)

Given an Echo State Network, train it on the target sequence y_target and return the optimised output weights Wₒᵤₜ.
"""
function train_esn!(esn, y_target, ridge_param)
    training_method = StandardRidge(ridge_param)
    return train(esn, y_target, training_method)
end


"""
Hyperparameters for an Echo State Network.
"""
struct ESNHyperparameters
    reservoir_size
    spectral_radius
    sparsity
    input_scale
    ridge_param
end


function precipitation_processing()
    precipitationDaily = zeros(0)
    for year in 1979:2019
        filename = "daily_precip_southern_amazonia/" * string(year) * ".csv"
        yearData = Matrix(CSV.read(filename,DataFrame))[:,1]
        precipitationDaily = vcat(precipitationDaily,yearData)
    end

    function leftMean(array::Vector{Float64},window::Int64)
        return [sum(array[x:x+window-1])/window for x in 1:size(array)[1]-window]
    end

    window = 10
    return leftMean(precipitationDaily,window)
end

function proximity_function() 
    proximityDaily = zeros(0)
    filename = "onset_retreat_length/ws_onset.csv"
    ws_onset = Matrix(CSV.read(filename,DataFrame, header = false))[:,1]
    filename = "onset_retreat_length/ds_onset.csv"
    ds_onset = Matrix(CSV.read(filename,DataFrame, header = false))[:,1]
    first_ws_onset = ws_onset[end]
    first_ws_length = 365 - first_ws_onset + ds_onset[1]
    proximityDaily = [1 - (x + (365 - first_ws_onset))/first_ws_length for x in 1:ds_onset[1]]
    last_ds_onset = ds_onset[1]
    
    for year in 1979:2019
        if year in [1980,1984,1988,1992,1996,2000,2004,2008,2012,2016]
            days = 366
        else
            days = 365
        end
        ds_length = ws_onset[year - 1979 + 1] - ds_onset[year - 1979 + 1]
        proximityDaily = vcat(proximityDaily,[x/ds_length for x in 1:ds_length])
        if year<2019
            ws_length = days - ws_onset[year - 1979 + 1] + ds_onset[year - 1979 + 1 + 1]
            proximityDaily = vcat(proximityDaily,[1 - x/ws_length for x in 1:ws_length])
        end
    end
    ws_length = 365 - ws_onset[41] + last_ds_onset
    return proximityDaily = vcat(proximityDaily,[1-x/ws_length for x in 1:365-ws_onset[41]])

end


function yearly_cycle(array)

    len = size(array)
    t = range(0,len[1])
    return cos.(2*π*(t.-152.25)/365.25)

end



function split_data(meanPrecipitationDaily, cosine_signal, proximityDaily, window) 
    print(size(meanPrecipitationDaily), size(cosine_signal[window+1:end-1]))
    data_x = hcat(meanPrecipitationDaily, cosine_signal[window+1:end-1])'
    data_y = proximityDaily[window+1:end]
    test_size = 15*365+4
    val_size = 5*365+2
    dt = 1
    
    # Split precipitation data
    train_x = data_x[:,1:test_size]
    val_x = data_x[:,test_size+1:test_size+val_size]
    test_x = data_x[:,test_size+val_size+1:end]
    
    # Split proximity data
    train_y = data_y[1:test_size]
    val_y = data_y[test_size+1:test_size+val_size]
    test_y = data_y[test_size+val_size+1:end]

    return train_x, val_x, test_x, train_y, val_y, test_y

end


param_grid = ESNHyperparameters[]  # Empty list of objects of type ESNHyperparameters

reservoir_sizes = [256, 512, 1024]
spectral_radii = [0.8, 1.0, 1.2]
sparsities = [0.03, 0.05]
input_scales = [0.1]
ridge_values = [0.0,10^(-10),10^(-8)]  # No noise so OLS is fine

# Take the Cartesian product of the possible hyperparameter values
for params in Iterators.product(reservoir_sizes, spectral_radii, sparsities, input_scales, ridge_values)
    push!(param_grid, ESNHyperparameters(params...))
end

println("$(length(param_grid)) hyperparameter combinations.")


"""
    cross_validate_esn(train_data, val_data, param_grid)

Do a grid search on the given param_grid to find the optimal hyperparameters.
"""
function cross_validate_esn(train_x, val_x, train_y, val_y, param_grid)
    best_loss = Inf
    best_params = nothing

    # We want
    x_train = train_x
    y_train = Array(train_y')
        
    for hyperparams in param_grid        
        # Unpack the hyperparameter struct
        (;reservoir_size, spectral_radius, sparsity, input_scale, ridge_param) = hyperparams

        loss_i = zeros(10)
        for i in 1:10
            # Generate and train an ESN
            esn = generate_esn(x_train, reservoir_size, spectral_radius, sparsity, input_scale)
            Wₒᵤₜ = train_esn!(esn, y_train, ridge_param)

            # Evaluate the loss on the validation set
            steps_to_predict = size(val_y,1)
            #prediction = esn(Generative(steps_to_predict), Wₒᵤₜ)
            prediction = esn(Predictive(val_x), Wₒᵤₜ)
            loss_i[i] = sum(abs2, prediction .- Array(val_y'))
        end
        
        # Mean loss
        loss = mean(loss_i)
        
        println("Hyperparameters: $(hyperparams)")
        println("Validation loss = ", @sprintf "%.1e" loss)
        
        
        # Keep track of the best hyperparameter values
        if loss < best_loss
            best_loss = loss
            best_params = hyperparams
        end
    end
    
    println("Optimal hyperparameters: $(best_params)")
    println("Validation loss = ", @sprintf "%.1e" best_loss)
    
    # Retrain the model using the optimal hyperparameters on both the training and validation data
    # This is necessary because we don't want errors accumulated during validation to affect the test error
    (; reservoir_size, spectral_radius, sparsity, input_scale, ridge_param) = best_params
    x = hcat(x_train, val_x) # Check if doesn't work, do array and ' stuff
    y = hcat(y_train, Array(val_y'))
    esn = generate_esn(x, reservoir_size, spectral_radius, sparsity, input_scale)
    Wₒᵤₜ = train_esn!(esn, y, ridge_param)
    
    return esn, Wₒᵤₜ
end

proximity_daily =  proximity_function()
mean_precipitation = precipitation_processing()
cosine_signal = yearly_cycle(proximity_daily)
train_x, val_x, test_x, train_y, val_y, test_y = split_data(mean_precipitation, cosine_signal, proximity_daily, 10)



#esn = generate_esn(x_train, reservoir_size, spectral_radius, sparsity, input_scale)
#Wₒᵤₜ = train_esn!(esn, y_train, ridge_param)

#prediction = esn(Predictive(val_x), Wₒᵤₜ)


#f(x, reservoir_sizes, spectral_radii, sparsities, input_scales, ridge_values) =  sum(abs2, x .- Array(val_y'))
function f_optim(reservoir_sizes, spectral_radii, sparsities, input_scales, ridge_values)
    x_train = train_x
    y_train = Array(train_y')
    loss_i = zeros(10)
    for i in 1:10
        # Generate and train an ESN
        esn = generate_esn(x_train, reservoir_sizes, spectral_radii, sparsities, input_scales)
        Wₒᵤₜ = train_esn!(esn, y_train, ridge_values)

        # Evaluate the loss on the validation set
        steps_to_predict = size(val_y,1)
        #prediction = esn(Generative(steps_to_predict), Wₒᵤₜ)
        prediction = esn(Predictive(val_x), Wₒᵤₜ)
        loss_i[i] = sum(abs2, prediction .- Array(val_y'))
    end
    
    # Mean loss
    loss = mean(loss_i)
end


ho = @hyperopt for i=50,
        sampler = RandomSampler(), # This is default if none provided
        reservoir_sizes = [256, 512, 1024],
        spectral_radii = [0.8, 1.0, 1.2],
        sparsities = [0.03, 0.05],
        ridge_values = [0.0,10^(-10),10^(-8)],
        input_scales = [0.1]

    print(i, "\t", reservoir_sizes, "\t", spectral_radii, "\t", sparsities, "\t", input_scales, "\t", ridge_values, "   \t")
    #esn = generate_esn(train_x, reservoir_sizes, spectral_radii, sparsities, input_scales)
    #Wₒᵤₜ = train_esn!(esn, Array(train_y'), ridge_values)

    @show f_optim(reservoir_sizes, spectral_radii, sparsities, input_scales, ridge_values)
end
best_params, min_f = ho.minimizer, ho.minimum
print(best_params, min_f)


#@time esn, Wₒᵤₜ = cross_validate_esn(train_x, val_x, train_y, val_y, param_grid);