using MKL
using DynamicalSystems       # For calculating Lyapunov exponents (a great package with much more to it than this)
using ReservoirComputing     # Julia implementation of reservoir computers
using Printf                 # Nice C-style string formatting
using Plots
using CSV
using DataFrames
using NaNStatistics
using Statistics
using JLD2
using StatsBase
using NCDatasets
using Dates


ENV["GKSwstype"]="nul"
clim_mean = 5.627757
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
window = 60

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

const DATA_DIRECTORY = "/cluster/projects/nn8008k/Nils/Monsoon_prediction/tropical_atlantic/"

"""
    load_yearly_sst(year::Int)

Load the sea surface temperature (SST) data for a specific year from a netCDF file.
"""
function load_yearly_sst(year::Int)
    filename = joinpath(DATA_DIRECTORY, "$(year)_STA.nc")
    ds = NCDatasets.Dataset(filename, "r")
    sst = ds["sst"][1, 1, 2:end]
    return sst
end

"""
    load_all_sst(start_year::Int, end_year::Int)

Load the SST data for all years in the range from start_year to end_year, inclusive.
The data for each year is concatenated into a single array.
"""
function load_all_sst(start_year::Int, end_year::Int)
    sst_data = [load_yearly_sst(year) for year in start_year:end_year]
    return vcat(sst_data...)
end

"""
    moving_average(array::Vector{Float64}, window::Int)

Compute the moving average of the input array with a specified window size.
The average is computed for each subarray of size `window` in the input array,
moving from left to right by one position at each step.
"""
function moving_average(array::Vector{T}, window::Int) where {T <: AbstractFloat}
    return [sum(array[i:i+window-1])/window for i in 1:(length(array)-window)]
end

"""
function moving_average(array::Vector{T}, window::Int) where {T <: AbstractFloat}
    # Validate the window size
    if window ≤ 0
        error("Window size must be positive.")
    end

    # Initialize an array to hold the moving averages
    ma = Vector{Float64}(undef, length(array) - window + 1)
    
    # Calculate the moving average for each position
    for i in 1:length(ma)
        ma[i] = mean(array[i:(i + window - 1)])
    end

    return ma
end
"""

"""
    load_sst()

Load SST data for all years from 1979 to 2019, remove any missing values,
and compute the moving average with a window size of 10.
"""
function load_sst()
    nta_daily = load_all_sst(1959, 2019)
    nta_daily = collect(skipmissing(nta_daily))

    return moving_average(nta_daily, window) .- mean(nta_daily)
end

function load_wind_patterns()

    filename = "sams_regimes_k9_pc100_n1000.csv"
    yearData = Matrix(CSV.read(filename,DataFrame))[365*9+2+window:365*9+2+22279] #1959 to 2019
end


function load_wind_patterns_anomaly()
    filename = "relative_climatology_CP8.csv"
    yearData = Matrix(CSV.read(filename,DataFrame))[365*19+5+window+1:365*19+5+22280] #1959 to 2019, 1940 to 1959 has 5 leap years
end

function precipitation_processing()
    precipitationDaily = zeros(0)
    for year in 1959:2019
        filename = "daily_precip_southern_amazonia/" * string(year) * ".csv"
        yearData = Matrix(CSV.read(filename,DataFrame))[1:end, 1] #.+ clim_mean
        precipitationDaily = vcat(precipitationDaily,yearData)
    end
    min_val = minimum(precipitationDaily)
    max_val = maximum(precipitationDaily)
    precipitationDaily = 1 .-  (precipitationDaily .- min_val) ./ (max_val - min_val)


    return moving_average(precipitationDaily,window)
end

#= function proximity_function() 
    proximityDaily = zeros(0)
    filename = "onset_retreat_length/WSL_onset_nofilter.csv"
    ws_onset = Matrix(CSV.read(filename, DataFrame, header = false))[:, 1] 
    filename = "onset_retreat_length/DSL_onset_nofilter.csv" 
    ds_onset = Matrix(CSV.read(filename, DataFrame, header = false))[:, 1]  
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
    return proximityDaily = vcat(proximityDaily, [1-x/ws_length for x in 1:365-ws_onset[61]])

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
    t = range(0,len[1]-1)
    return cos.(2*π*(t.-152.5)/365.25)

end





function load_olr()
    filename = "olr/ts_south_amazon.csv"
    olrDaily = Float64.(Matrix(CSV.read(filename,DataFrame))[1:end,3])
    olrDaily = olrDaily .- 240
    # Normalize the olrDaily array between 0 and 1
    #min_val = minimum(olrDaily)
    max_val = maximum(olrDaily)
    #olrDaily = (olrDaily .- min_val) ./ (max_val - min_val)
    olrDaily = olrDaily ./ max_val
    println("before moving avg.", size(olrDaily))
    println("after moving avg.", size(moving_average(olrDaily,window)))

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

param_grid = ESNHyperparameters[]  # Empty list of objects of type ESNHyperparameters


"""
reservoir_sizes = 1024#512
spectral_radii = 0.7#0.8
sparsities = 0.05#0.03
input_scales = 0.1
ridge_values = 10^(-8)  # No noise so OLS is fine
"""

proximity =  proximity_function()

proximity_daily = proximity[window+1:end]

#mean_precipitation = precipitation_processing()
olr_daily = load_olr()#[1:size(mean_precipitation, 1)]
#proximity_daily = proximity_daily[1:size(mean_precipitation, 1)]

#nta_daily = load_sst()
#wind_patterns = load_wind_patterns()
#plot(nta_daily, ylabel = "onset")
#range(1, size(preiction_mean[1,:])[1])

#savefig("plots/nta_detrended.png")

cosine_signal = yearly_cycle(proximity_daily)


nta_daily = load_sst()
wind_patterns = load_wind_patterns_anomaly()
println("size cosine: ", size(cosine_signal), "size olr daily: ", size(olr_daily), "size prox f: ", size(proximity_daily))
train_x, val_x, test_x, train_y, val_y, test_y = split_data(olr_daily, wind_patterns, cosine_signal, proximity_daily, window)

function get_onset_and_retreatdate(year)
    ground_truth = select_year_data(test_y, year, 2000)
    mxval, mxindx = findmax(ground_truth)
    minval, minindx = findmin(ground_truth)

    return minindx, mxindx
end

function ensemble_prediction(leadtime)
    prediction_test = zeros(100, size(test_x)[2])
    for i = 1:100
        esn = jldopen("esn_states/esn_olr_$i.jld2")["esn_1"]
        Wₒᵤₜ =jldopen("W_Matrix/W_olr_$i.jld2")["Wₒᵤₜ_1"]
        prediction_test[i, :] = esn(Predictive(test_x), Wₒᵤₜ)
       
    end
    prediction_mean = mean(prediction_test, dims = 1)[1,:]
    #println(prediction_mean)
    #label = ["actual" "predicted"]
    
    println(size(test_y), size(prediction_mean[1]), size(prediction_mean))
    regr_1 =  zeros(21)
    regr_2 = zeros(21)
   
    for i=1:21
        start_date, end_date = get_onset_and_retreatdate(2000 + i)
        start_date = start_date + 10# window # onset date + 10 days 
        end_date = end_date - leadtime 
        x_regression = start_date:end_date
        X = hcat(ones(end_date - start_date + 1), x_regression)
        println("X shape: ", size(X))
        start_indices_for_year, end_indices_for_year = get_indices_for_year_range(2000+ i, 2000+i) .- get_indices_for_year_range(2000, 2001)[1]
        println(get_indices_for_year_range(2000, 2001)[1])
        println(start_indices_for_year, " ", end_indices_for_year)
        y_regression = prediction_mean[start_date+start_indices_for_year:end_date+start_indices_for_year]
        println(size(y_regression))
        regr_1[i], regr_2[i] =  X\y_regression
        plot(select_year_data(test_y, 2000 + i, 2000), ylabel = "onset")
        plot!(range(start_date, 365), regr_1[i] .+ regr_2[i] .* range(start_date, 365))
        mxval, mxindx = findmax(test_x[1,:])
        #println(mxval)
        plot!(select_year_data(test_x[1,:]./mxval, 2000 + i, 2000))
        
        vline!([start_date], color="red", label ="start date")
        println(findfirst(x -> x >= 1, (regr_1[i] .+ regr_2[i] .* range(start_date, 365)))+start_date)
        vline!([findfirst(x -> x >= 1, (regr_1[i] .+ regr_2[i] .* range(start_date, 365))) + start_date], color="green", label ="onset")
        vline!([end_date], color="orange", label="end date")
        for j=1:100
            plot!(select_year_data(prediction_test[j, :], 2000 + i, 2000), alpha=0.5, lw=0.1, label = "")
        end
        #range(1, size(preiction_mean[1,:])[1])
        plot!(select_year_data(prediction_mean, i+2000, 2000), dpi = 600, lw=2, color="black", label ="Mean prediction")
        savefig("plots/mean_prediction_year_olr_$i.png")
    end



    return prediction_mean

end

function linear_regr_pred(y_array, leadtime)
    filename = "onset_retreat_length/onset_dates_olr.csv"
    ws_onset = Matrix(CSV.read(filename,DataFrame, header = false))[:,1]
    

    prediction = zeros(21)
    regr_1 =  zeros(21)
    regr_2 = zeros(21)
    for i=1:21
        start_date, end_date = get_onset_and_retreatdate(2000 + i)
        start_date = start_date + 10 #window # onset date + 10 days 
        end_date = end_date - leadtime 
        x_regression = start_date:end_date
        X = hcat(ones(end_date - start_date + 1), x_regression)
        start_indices_for_year, end_indices_for_year = get_indices_for_year_range(2000+ i, 2000+i) .- get_indices_for_year_range(2000, 2001)[1]
        y_regression = y_array[start_date+start_indices_for_year:end_date+start_indices_for_year]
        regr_1[i], regr_2[i] =  X\y_regression
        #print(regr_1[i] .+ regr_2[i] .* range(start_date, 365))
        println((regr_1[i] .+ regr_2[i] .* range(start_date, 400)))
        prediction[i] = findfirst(x -> x >= 1, (regr_1[i] .+ regr_2[i] .* range(start_date, 365))) + start_date 

    end
    println(size(prediction), size(ws_onset[end-19:end]))
    enddate = zeros(21)
    for i =1:21
        start_date, enddate[i] = get_onset_and_retreatdate(2000 + i) #.+ window
    end
    println("rmse: ", rmsd(prediction, enddate; normalize=false))
    println("rmse, mean: ", rmsd(repeat([mean(enddate)], length(enddate)), enddate; normalize=false))
    
    
    label = "predicted"
    x_label = range(2001, 2021)
    plot(x_label, prediction, xlabel = "Year AD", label = label, ylabel = "Wet Season Onset")
    println(size(ws_onset))
    println(size(ws_onset[end-19:end]))
    plot!(x_label, enddate, ribbon=(7,7), label = "actual", show = true)
    savefig("plots/test_set_olr_prediction.png")

end

prediction_mean = ensemble_prediction(60)
linear_regr_pred(prediction_mean, 60)