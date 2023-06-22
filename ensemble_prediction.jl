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

ENV["GKSwstype"]="nul"

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

const DATA_DIRECTORY = "/cluster/projects/nn8008k/Nils/Monsoon_prediction/tropical_atlantic/"

"""
    load_yearly_sst(year::Int)

Load the sea surface temperature (SST) data for a specific year from a netCDF file.
"""
function load_yearly_sst(year::Int)
    filename = joinpath(DATA_DIRECTORY, "$(year)_NTA.nc")
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
    return [sum(array[i:i+window-1])/window for i in 1:length(array)-window]
end


"""
    load_sst()

Load SST data for all years from 1979 to 2019, remove any missing values,
and compute the moving average with a window size of 10.
"""
function load_sst()
    nta_daily = load_all_sst(1979, 2019)
    nta_daily = collect(skipmissing(nta_daily))
    window = 10
    return moving_average(nta_daily, window) .- mean(nta_daily)
end



function precipitation_processing()
    precipitationDaily = zeros(0)
    for year in 1979:2019
        filename = "daily_precip_southern_amazonia/" * string(year) * ".csv"
        yearData = Matrix(CSV.read(filename,DataFrame))[:,1]
        precipitationDaily = vcat(precipitationDaily,yearData)
    end

    window = 10
    return moving_average(precipitationDaily,window)
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



function split_data(meanPrecipitationDaily, nta_daily, cosine_signal, proximityDaily, window) 
    #print(size(meanPrecipitationDaily), size(cosine_signal[window+1:end-1]))
    data_x = hcat(meanPrecipitationDaily, nta_daily, cosine_signal[window+1:end-1])'
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

reservoir_sizes = 512
spectral_radii = 0.8
sparsities = 0.03
input_scales = 0.1
ridge_values = 10^(-8)  # No noise so OLS is fine

proximity_daily =  proximity_function()
mean_precipitation = precipitation_processing()

nta_daily = load_sst()

cosine_signal = yearly_cycle(proximity_daily)
train_x, val_x, test_x, train_y, val_y, test_y = split_data(mean_precipitation, nta_daily, cosine_signal, proximity_daily, 10)


function ensemble_prediction()
    prediction_test = zeros(100, size(test_x)[2])
    for i = 1:100
        esn = jldopen("esn_states/esn_$i.jld2")["esn"]
        Wₒᵤₜ =jldopen("W_Matrix/W_$i.jld2")["Wₒᵤₜ"]
        prediction_test[i, :] = esn(Predictive(test_x), Wₒᵤₜ)
    end
    prediction_mean = mean(prediction_test, dims = 1)[1,:]
    #println(prediction_mean)
    #label = ["actual" "predicted"]
    
    println(size(test_y), size(prediction_mean[1]), size(prediction_mean))

    plot(test_y, ylabel = "onset")
    #range(1, size(preiction_mean[1,:])[1])
    plot!(prediction_mean, dpi = 600)
    savefig("plots/mean_prediction.png")

    return prediction_mean

end

function linear_regr_pred(y_array, start_date, end_date)
    filename = "onset_retreat_length/ws_onset.csv"
    ws_onset = Matrix(CSV.read(filename,DataFrame, header = false))[:,1]

    x_regression = start_date:end_date
    X = hcat(ones(end_date - start_date + 1), x_regression)
    prediction = zeros(20)
    regr_1 =  zeros(20)
    regr_2 = zeros(20)
    for i=1:20
        y_regression = y_array[start_date+(i*365):end_date+(i*365)]
        regr_1[i], regr_2[i] =  X\y_regression
        prediction[i] = findall(x -> x >= 1, (regr_1[i] .+ regr_2[i] .* range(start_date, 365)))[1] + start_date
    end
    println(size(prediction), size(ws_onset[end-19:end]))
    println("rmse: ", rmsd(prediction, ws_onset[end-19:end]; normalize=false))
    
    
    label = "predicted"
    x_label = range(2001, 2020)
    plot(x_label, prediction, xlabel = "Year AD", label = label, ylabel = "Wet Season Onset")
    plot!(x_label, ws_onset[end-19:end], ribbon=(5,5), label = "actual", show = true)
    savefig("plots/test_set_prediction.png")

end

prediction_mean = ensemble_prediction()
linear_regr_pred(prediction_mean, 100, 220)