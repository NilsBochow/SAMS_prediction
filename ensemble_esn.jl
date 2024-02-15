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
using StatsBase
using Dates
#using JLSO

println("Number of Threads:", Threads.nthreads())
a = 0.1
Wᵢₙ = WeightedLayer(scaling = a)
clim_mean = 5.627757
window = 60
function generate_esn(
    input_signal,
    reservoir_size = 256,
    spectral_radius = 0.7,
    sparsity = 0.04,
    input_scale = 0.4,
)
    Wᵢₙ = WeightedLayer(scaling = input_scale)
    W = RandSparseReservoir(reservoir_size, radius = spectral_radius, sparsity = sparsity)
    return ESN(input_signal, reservoir = W, input_layer = Wᵢₙ, bias=0.01)
end

ridge_param = 0#1e-8 # OLS
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
    return [sum(array[i:i+window-1])/window for i in 1:length(array)-window]
end


"""
    load_sst()

Load SST data for all years from 1959 to 2019, remove any missing values,
and compute the moving average with a window size of 10.
"""
function load_sst()
    nta_daily = load_all_sst(1959, 2019)
    nta_daily = collect(skipmissing(nta_daily))
    return  moving_average(nta_daily, window) .- mean(nta_daily)
end

function load_wind_patterns()
    filename = "sams_regimes_k9_pc100_n1000.csv"
    yearData = Matrix(CSV.read(filename,DataFrame))[365*9+2+window+1:365*9+2+22280] #1959 to 2019
end

function load_wind_patterns_anomaly()
    filename = "relative_climatology_CP8.csv"
    yearData = Matrix(CSV.read(filename,DataFrame))[365*19+5+window+1:365*19+5+22280] #1959 to 2019, 1940 to 1959 has 5 leap years
end


function precipitation_processing()
    precipitationDaily = zeros(0)
    for year in 1959:2019
        filename = "daily_precip_southern_amazonia/" * string(year) * ".csv"
        yearData = Matrix(CSV.read(filename,DataFrame))[:,1] #.- clim_mean

        precipitationDaily = vcat(precipitationDaily,yearData) 
    end
    min_val = minimum(precipitationDaily)
    max_val = maximum(precipitationDaily)
    precipitationDaily = 1 .-  (precipitationDaily .- min_val) ./ (max_val - min_val)

    return moving_average(precipitationDaily,window)
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


"""
function split_data(proxyData, mean_precipitation, wind_patterns, cosine_signal, proximityDaily, window) 
    print("split data", size(proxyData), size(wind_patterns), size(cosine_signal[window+1:end]), size(mean_precipitation), size(proximityDaily))
    proxyData = proxyData #.* mean_precipitation
    data_x = hcat(proxyData, cosine_signal[window+1:end])' #, wind_patterns)'
    data_y = proximityDaily[window+1:end]
    train_size = 30*365+9 #leap years
    val_size = 10*365+2
    dt = 1
    
    # Split precipitation data
    train_x = data_x[:,1:train_size]
    val_x = data_x[:,train_size+1:train_size+val_size]
    test_x = data_x[:,train_size+val_size+1:end]
    
    # Split proximity data
    train_y = data_y[1:train_size]
    val_y = data_y[train_size+1:train_size+val_size]
    test_y = data_y[train_size+val_size+1:end]

    return train_x, val_x, test_x, train_y, val_y, test_y

end
"""

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

reservoir_sizes = 1024
spectral_radii = 1.0
sparsities = 0.05
input_scales = 0.04
ridge_values = 10^(-8) #0 #10^(-8)  # No noise so OLS is fine


## mean precip has the size 22260 (olr and proximity are longer since they go until year 2021)
proximity =  proximity_function()

proximity_daily = proximity[window+1:end]
mean_precipitation = precipitation_processing()
olr_daily = load_olr()


#nta_daily = load_sst()
wind_patterns = load_wind_patterns_anomaly()
#plot(nta_daily, ylabel = "onset")
#range(1, size(preiction_mean[1,:])[1])

#savefig("plots/nta_detrended.png")

cosine_signal = yearly_cycle(proximity_daily)
println("olr shape", size(olr_daily))
println("proximity shape", size(proximity_daily))
println("mean_precipitation shape", size(mean_precipitation))
println("cosine_signal shape", size(cosine_signal))

train_x, val_x, test_x, train_y, val_y, test_y = split_data(olr_daily, wind_patterns, cosine_signal, proximity_daily, window)

function test_ensemble()
    acc = Atomic{Int64}(0)

end

function ensemble_simulation(reservoir_sizes, spectral_radii, sparsities, input_scales, ridge_values)
    x_train = train_x
    y_train = Array(train_y')
    loss_100 = zeros(100)
    acc = Atomic{Int64}(0)
    number_simulations = 100000
    esn = Dict{Int, Any}()

    Wₒᵤₜ = Dict{Int, Any}()
    prediction = Dict{Int, Any}()
    loss = Dict{Int, Any}()
    sl = ReentrantLock()

    Threads.@threads for i in 1:100
        # Test if actually 100 000 simulations
        atomic_add!(acc, 1)
        # Generate and train an ESN
        esn[Threads.threadid()] = generate_esn(x_train, reservoir_sizes, spectral_radii, sparsities, input_scales)
        Wₒᵤₜ[Threads.threadid()] = Dict{Symbol, Any}()  # Initialize Wₒᵤₜ dictionary
        prediction[Threads.threadid()] = Dict{Symbol, Any}()  # Initialize prediction dictionary
        loss[Threads.threadid()] = Dict{Symbol, Any}()  # Initialize loss dictionary
        Wₒᵤₜ[Threads.threadid()] = train_esn!(esn[Threads.threadid()], y_train, ridge_values)
        prediction[Threads.threadid()] = esn[Threads.threadid()](Predictive(val_x), Wₒᵤₜ[Threads.threadid()])
        loss[Threads.threadid()] = rmsd(prediction[Threads.threadid()], Array(val_y'))
        #loss[Threads.threadid()] = sum(abs2, prediction[Threads.threadid()] .- Array(val_y')) + ridge_param*(tr(Wₒᵤₜ[Threads.threadid()]*transpose(Wₒᵤₜ[Threads.threadid()])))
        loss_100[i] = loss[Threads.threadid()]
        println(i, " ", loss[Threads.threadid()])
        Threads.lock(sl) do
            Wₒᵤₜ_1 = Wₒᵤₜ[Threads.threadid()]
            esn_1 =  esn[Threads.threadid()]
            jldsave("W_Matrix/W_olr_$i.jld2"; Wₒᵤₜ_1)
            jldsave("esn_states/esn_olr_$i.jld2"; esn_1)
        end
    end
    Threads.@threads for i in 100:number_simulations
        # Test if actually 100 000 simulations
        atomic_add!(acc, 1)
        # Generate and train an ESN
        esn[Threads.threadid()] = generate_esn(x_train, reservoir_sizes, spectral_radii, sparsities, input_scales)
        Wₒᵤₜ[Threads.threadid()] = Dict{Symbol, Any}()  # Initialize Wₒᵤₜ dictionary
        prediction[Threads.threadid()] = Dict{Symbol, Any}()  # Initialize prediction dictionary
        loss[Threads.threadid()] = Dict{Symbol, Any}()  # Initialize loss dictionary
        Wₒᵤₜ[Threads.threadid()] = train_esn!(esn[Threads.threadid()], y_train, ridge_values)
        prediction[Threads.threadid()] = esn[Threads.threadid()](Predictive(val_x), Wₒᵤₜ[Threads.threadid()])
        loss[Threads.threadid()] = rmsd(prediction[Threads.threadid()], Array(val_y'))
        println(Threads.threadid())
        """if i <= 200
            loss_100[i] = loss[Threads.threadid()]
            println(loss[Threads.threadid()])
            Threads.lock(sl) do
                println(Threads.threadid())
                Wₒᵤₜ_1 = Wₒᵤₜ[Threads.threadid()]
                esn_1 =  esn[Threads.threadid()]
                jldsave("W_Matrix/W_30d_$i.jld2"; Wₒᵤₜ_1)
                jldsave("esn_states/esn_30d_$i.jld2"; esn_1)
            end"""
        #else 
        if any(loss_100 .> loss[Threads.threadid()])
            Threads.lock(sl) do
                mxval, mxindx = findmax(loss_100)
                if loss[Threads.threadid()]<loss_100[mxindx]
                    println("Found better loss: ", loss[Threads.threadid()], " Index: ", i, "Old loss: ", mxval, " Old Index: ", mxindx)
                    loss_100[mxindx] = loss[Threads.threadid()]
                    Wₒᵤₜ_1 = Wₒᵤₜ[Threads.threadid()]
                    esn_1 =  esn[Threads.threadid()]
                    println(Threads.threadid())
                    jldsave("W_Matrix/W_olr_$mxindx.jld2"; Wₒᵤₜ_1)
                    jldsave("esn_states/esn_olr_$mxindx.jld2"; esn_1)
                else
                    nothing
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