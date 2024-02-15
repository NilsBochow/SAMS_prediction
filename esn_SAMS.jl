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
using StatsBase
using Dates 

a = 0.1
Wᵢₙ = WeightedLayer(scaling = a)
window=30
reservoir_size = 1000           # Number of nodes in the reservoir
spectral_radius = 1.0           # Good starting value
sparsity = 10 / reservoir_size  # Each node is connected to 10 others, on average

W = RandSparseReservoir(reservoir_size, radius = spectral_radius, sparsity = sparsity)
function moving_average(array::Vector{T}, window::Int) where {T <: AbstractFloat}
    return [sum(array[i:i+window-1])/window for i in 1:length(array)-window]
end

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
    return ESN(input_signal, reservoir = W, input_layer = Wᵢₙ, bias=0.01)
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


function load_wind_patterns()
    window = 30
    filename = "sams_regimes_k9_pc100_n1000.csv"
    yearData = Matrix(CSV.read(filename,DataFrame))[365*9+2+window+1:365*9+2+22280] #1959 to 2019
end


function precipitation_processing()
    precipitationDaily = zeros(0)
    for year in 1959:2019
        filename = "daily_precip_southern_amazonia/" * string(year) * ".csv"
        yearData = Matrix(CSV.read(filename,DataFrame))[1:end,1]
        print(size(yearData))
        precipitationDaily = vcat(precipitationDaily,yearData)
    end

    window = 30
    return moving_average(precipitationDaily,window)
end


#= function proximity_function() 
    proximityDaily = zeros(0)
    filename = "onset_retreat_length/WSL_onset_2.csv"
    ws_onset = Matrix(CSV.read(filename,DataFrame, header = false))[:,1] .*5
    filename = "onset_retreat_length/DSL_onset_2.csv"
    ds_onset = Matrix(CSV.read(filename,DataFrame, header = false))[:,1] .*5
    first_ws_onset = ws_onset[end]
    first_ws_length = 365 - first_ws_onset + ds_onset[1]
    proximityDaily = [1 - (x + (365 - first_ws_onset))/first_ws_length for x in 1:ds_onset[1]]
    last_ds_onset = ds_onset[1]
    
    for year in 1959:2019
        if year in [1960,1964,1968,1972,1976,1980,1984,1988,1992,1996,2000,2004,2008,2012,2016]
            days = 366
        else
            days = 365
        end
        ds_length = ws_onset[year - 1959 + 1] - ds_onset[year - 1959 + 1]
        proximityDaily = vcat(proximityDaily,[x/ds_length for x in 1:ds_length])
        if year<2019
            ws_length = days - ws_onset[year - 1959 + 1] + ds_onset[year - 1959 + 1 + 1]
            proximityDaily = vcat(proximityDaily,[1 - x/ws_length for x in 1:ws_length])
        end
    end
    ws_length = 365 - ws_onset[61] + last_ds_onset
    return proximityDaily = vcat(proximityDaily,[1-x/ws_length for x in 1:365-ws_onset[61]])

end =#


function proximity_function() 
    proximityDaily = zeros(0)
    filename = "onset_retreat_length/retreat_ds_olr_30d.csv"#onset_dates_olr.csv"
    ws_onset = Matrix(CSV.read(filename,DataFrame, header = false))[:,1]
    filename = "onset_retreat_length/onset_ds_olr_30d.csv"#retreat_dates_olr.csv"
    ds_onset = Matrix(CSV.read(filename,DataFrame, header = false))[:,1]
    println("ds onset length:", size(ds_onset))
    first_ws_onset = ws_onset[1] #ws_onset[end]
    first_ws_length = 365 - first_ws_onset + ds_onset[1]
    println("first ws onset: ", first_ws_onset)
    println("first ds onset: ", ds_onset[1])
    proximityDaily = [1 - (x + (365 - first_ws_onset))/first_ws_length for x in 1:ds_onset[1]]
    
    mxval, mxindx = findmax(proximityDaily)
    minval, minindx = findmin(proximityDaily)
    println("min/max first year: ", minindx, " ", mxindx)
    last_ds_onset = ds_onset[end]
    for year in 1959:2021
        if isleapyear(year)
            days = 366
        else
            days = 365
        end
        ds_length = ws_onset[year - 1959 + 1] - ds_onset[year - 1959 + 1]
        proximityDaily = vcat(proximityDaily,[x/ds_length for x in 1:ds_length])

        if year<2021
            ws_length = days - ws_onset[year - 1959 + 1] + ds_onset[year - 1959 + 1 + 1]
            proximityDaily = vcat(proximityDaily,[1 - x/ws_length for x in 1:ws_length])

        end
    end
    println("proximity size", size(proximityDaily))
    ws_length = 365 - ws_onset[63] + last_ds_onset
    proximityDaily = vcat(proximityDaily, [1-x/ws_length for x in 1:365-ws_onset[63]])
    println("proximity size", size(proximityDaily))
    return proximityDaily
end



function yearly_cycle(array)

    len = size(array)
    t = range(0,len[1]-1 )
    return cos.(2*π*(t.-152.5)/365.25)

end

function load_wind_patterns_anomaly()
    filename = "relative_climatology_CP8.csv"
    yearData = Matrix(CSV.read(filename,DataFrame))[365*19+5+window+1:365*19+5+22280] #1959 to 2019, 1940 to 1959 has 5 leap years
end


function load_olr()
    filename = "olr/ts_south_amazon.csv"
    olrDaily = Float64.(Matrix(CSV.read(filename,DataFrame))[1:end,3])
    olrDaily = olrDaily .- 240
    # Normalize the olrDaily array between 0 and 1
    #min_val = minimum(olrDaily)
    max_val = maximum(olrDaily)
    #olrDaily = (olrDaily .- min_val) ./ (max_val - min_val)
    olrDaily = olrDaily./ max_val
    println("before moving avg.", size(olrDaily))

    return moving_average(olrDaily,window)
end
function get_indices_for_year_range(start_year::Int, end_year::Int)
    # Validate the input years
    if start_year < 1959 || end_year > 2022 || start_year > end_year
        error("Invalid year range. Start year must be >= 1959, end year <= 2022, and start year <= end year.")
    end

    # Helper function to calculate the number of days since 1959 to the beginning of a given year
    days_since_1959_to_start_of_year = year -> sum([isleapyear(y) ? 366 : 365 for y in 1959:(year - 1)])

    # Calculate the start index for the start year
    start_index = days_since_1959_to_start_of_year(start_year) + 1

    # Calculate the end index for the end year
    # We add the days in the end year to the days up to the start of the end year
    end_index = days_since_1959_to_start_of_year(end_year) + (isleapyear(end_year) ? 366 : 365)

    return start_index, end_index
end


function split_data(proxyData, wind_patterns, cosine_signal, proximityDaily, window) 
   
    proxyData = proxyData #.* mean_precipitation
    data_x = hcat(proxyData, cosine_signal)'#, wind_patterns)'
    data_y = proximityDaily#[window+1:end]
    train_start_index, train_end_index = get_indices_for_year_range(1959,1989)
    train_end_index = train_end_index - window
    val_start_index, val_end_index = get_indices_for_year_range(1990,2000) 
    val_start_index = val_start_index - window
    val_end_index = val_end_index - window
    test_start_index, test_end_index = get_indices_for_year_range(2000,2021)
    test_start_index = test_start_index - window
    test_end_index = test_end_index - window
    train_size = 30*365+9 #leap years
    val_size = 10*365+2
    dt = 1
    
    # Split precipitation data
    train_x = data_x[:,1:train_end_index]
    val_x = data_x[:,val_start_index:val_end_index]
    test_x = data_x[:,test_start_index:test_end_index]
    
    # Split proximity data
    train_y = data_y[1:train_end_index]
    val_y = data_y[val_start_index:val_end_index]
    test_y = data_y[test_start_index:test_end_index]

    return train_x, val_x, test_x, train_y, val_y, test_y

end
function select_year_data(daily_data::Vector, year::Int, start_year::Int)
    # Check if the year is within the valid range
    if year < 1959 || year > 2022
        error("Year must be between 1959 and 2022")
    end

    # Calculate the number of days from 1959 to the beginning of the input year
    #start_year = 1959
    days_before_year = sum([isleapyear(y) ? 366 : 365 for y in start_year:(year - 1)])

    # Calculate the number of days in the input year
    days_in_year = isleapyear(year) ? 366 : 365

    # Select the data for the input year
    start_index = days_before_year + 1
    end_index = start_index + days_in_year - 1

    # Return the selected data
    return daily_data[start_index:end_index]
end

proximity_daily =  proximity_function()[window+1:end]
mean_precipitation = precipitation_processing()
olr_daily = load_olr()
proximity_daily = proximity_daily
wind_patterns = load_wind_patterns_anomaly()
cosine_signal = yearly_cycle(proximity_daily)
println("olr shape", size(olr_daily))
println("proximity shape", size(proximity_daily))
println("mean_precipitation shape", size(mean_precipitation))
println("cosine_signal shape", size(cosine_signal))
train_x, val_x, test_x, train_y, val_y, test_y = split_data(olr_daily, wind_patterns, cosine_signal, proximity_daily, window)

"""
f_optim(reservoir_sizes, spectral_radii, sparsities, input_scales, ridge_values)

Function to optimize. Calculates and returns the mean loss of 10 ESNs with the same hyperparameter values on the validation set.
"""
function f_optim(reservoir_sizes, spectral_radii, sparsities, input_scales, ridge_values)
    x_train = train_x
    y_train = Array(train_y')
    loss_i = zeros(50)
    for i in 1:50
        # Generate and train an ESN
        esn = generate_esn(x_train, reservoir_sizes, spectral_radii, sparsities, input_scales)
        Wₒᵤₜ = train_esn!(esn, y_train, ridge_values)

        # Evaluate the loss on the validation set
        steps_to_predict = size(val_y,1)
        prediction = esn(Predictive(val_x), Wₒᵤₜ)
        loss_i[i] = rmsd(prediction, Array(val_y'))
    end
    
    # Mean loss
    loss = mean(loss_i)
end

"""
Optimizing the Hyperparameters using the Hyperopt.jl package.
"""
ho = @hyperopt for i=100,
        sampler = RandomSampler(), # This is default if none provided
        reservoir_sizes = [100, 256, 512, 1024, 2048],
        spectral_radii = [0.7, 0.8, 0.9, 1.0, 1.1],
        sparsities = [0.03, 0.04, 0.05],
        ridge_values = [0.0,10^(-10),10^(-8)],
        input_scales = [0.1,0.04]

    print(i, "\t", reservoir_sizes, "\t", spectral_radii, "\t", sparsities, "\t", input_scales, "\t", ridge_values, "   \t")

    @show f_optim(reservoir_sizes, spectral_radii, sparsities, input_scales, ridge_values)
end
best_params, min_f = ho.minimizer, ho.minimum
print(best_params, min_f)

