using MKL
using DifferentialEquations  # For generating synthetic training data
using DynamicalSystems       # For calculating Lyapunov exponents (a great package with much more to it than this)
using ReservoirComputing     # Julia implementation of reservoir computers
using Printf                 # Nice C-style string formatting
using Plots
using CSV
using DataFrames
using NaNStatistics
using Statistics
using Base.Threads
using JLD2
using NCDatasets
#using JLSO

println("Number of Threads:", Threads.nthreads())
a = 0.1
Wᵢₙ = WeightedLayer(scaling = a)


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
    return moving_average(nta_daily, window)
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
"""
function load_sst()
    nta_daily = zeros(0)
    for year in 1979:2019
        filename = "/cluster/projects/nn8008k/Nils/Monsoon_prediction/tropical_atlantic/" * string(year) * "_NTA.nc"
        ds = NCDatasets.Dataset(filename,"r") 
        nta = ds["sst"][1, 1, 2:end]
        nta_daily = vcat(nta_daily,nta)

    end

    function leftMean(array::Vector{Float64},window::Int64)
        return [sum(array[x:x+window-1])/window for x in 1:size(array)[1]-window]
    end
    nta_daily = collect(skipmissing(nta_daily))

    window = 10
    return leftMean(nta_daily,window)
end
"""

subdata = load_sst()
println(size(subdata))
exit()

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
    #print(size(meanPrecipitationDaily), size(cosine_signal[window+1:end-1]))
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

reservoir_sizes = 512
spectral_radii = 0.8
sparsities = 0.03
input_scales = 0.1
ridge_values = 10^(-8)  # No noise so OLS is fine

proximity_daily =  proximity_function()
mean_precipitation = precipitation_processing()
cosine_signal = yearly_cycle(proximity_daily)
train_x, val_x, test_x, train_y, val_y, test_y = split_data(mean_precipitation, cosine_signal, proximity_daily, 10)


function test_ensemble()
    acc = Atomic{Int64}(0)

end

function ensemble_simulation(reservoir_sizes, spectral_radii, sparsities, input_scales, ridge_values)
    x_train = train_x
    y_train = Array(train_y')
    loss_100 = zeros(100)
    acc = Atomic{Int64}(0)
    number_simulations= 100000
    sl = ReentrantLock()
    Threads.@threads for i in 1:number_simulations
        # Test if actually 100 000 simulations
        atomic_add!(acc, 1)
        # Generate and train an ESN
        esn = generate_esn(x_train, reservoir_sizes, spectral_radii, sparsities, input_scales)
        Wₒᵤₜ = train_esn!(esn, y_train, ridge_values)
        prediction = esn(Predictive(val_x), Wₒᵤₜ)
        loss = sum(abs2, prediction .- Array(val_y'))
        
        if i <= 100
            loss_100[i] = loss
            println(loss)
            Threads.lock(sl) do
                jldsave("W_Matrix/W_$i.jld2"; Wₒᵤₜ)
                jldsave("esn_states/esn_$i.jld2"; esn)
            end
        else 
            if any(loss_100 .> loss)
                Threads.lock(sl) do
                println("Found better loss: ", loss, " Index: ", i)
                    mxval, mxindx = findmax(loss_100)
                    loss_100[mxindx] = loss
                    jldsave("W_Matrix/W_$mxindx.jld2"; Wₒᵤₜ)
                    jldsave("esn_states/esn_$mxindx.jld2"; esn)
                end
            end
        end
    end
    
    # Mean loss
    loss = mean(loss_100)
    println("Final mean loss: ", loss)
    println("Number of simulations: ", number_simulations)
    println("Actual number of simulations: ", acc[])

end


ensemble_simulation(reservoir_sizes, spectral_radii, sparsities, input_scales, ridge_values)