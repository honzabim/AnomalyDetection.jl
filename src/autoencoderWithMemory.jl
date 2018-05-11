push!(LOAD_PATH, "/home/jan/dev/OSL/KNNmemory")
using KNNmem

"""
	DEFAULT_LABEL - label to use for the first random fill of the memory
"""
const DEFAULT_LABEL = 0

"""
	AutoencoderWithMemory{encoder, decoder, memory}

Flux-like structure for the basic autoencoder.
"""

struct AutoencoderWithMemory
	encoder
	decoder
	memory::KNNmemory

	# TODO: describe all the parameters

	function AutoencoderWithMemory(esize::Array{Int64,1}, dsize::Array{Int64,1}, memorySize::Integer, keySize::Integer, k::Integer,
		 labelCount::Integer, α = 0.1, activation = Flux.relu, layer = ResDense)

		@assert size(esize, 1) >= 3
		@assert size(dsize, 1) >= 3
		@assert esize[end] == dsize[1]
		@assert esize[1] == dsize[end]

		# construct the encoder
		encoder = aelayerbuilder(esize, activation, layer)

		# construct the decoder
		decoder = aelayerbuilder(dsize, activation, layer)

		# finally construct the ae struct
		return new(encoder, decoder, KNNmemory(memorySize, keySize, k, labelCount, α))
	end
end

reconstructionError(ae::AutoencoderWithMemory, x) = Flux.mse(ae.decoder(ae.encoder(x)), x)
evalReconstructionError(ae::AutoencoderWithMemory, x) = println("Reconstruction error: ", Flux.Tracker.data(reconstructionError(ae, X)), "\n")
memoryTrainQuery(ae::AutoencoderWithMemory, data, label) = trainQuery!(ae.memory, ae.encoder(data), label)
learnAnomaly(ae::AutoencoderWithMemory, data, label = 1, γ = 0.5) = reconstructionError(ae, data) + γ * trainQuery!(ae.memory, ae.encoder(data), label)
memoryClassify(ae::AutoencoderWithMemory, x) = query(ae.memory, ae.encoder(x))

function learnRepresentation(ae::AutoencoderWithMemory, x)
    latentVariable = ae.encoder(x)
    trainQuery!(ae.memory, latentVariable, DEFAULT_LABEL)
    return Flux.mse(ae.decoder(latentVariable))
end

################
### training ###
################


"""
	fit!(ae, X, L, [iterations, cbit, verb, rdelta, tracked])

Trains the AE.
ae - AE type object
X - data array with instances as columns
L - batchsize
iterations - number of iterations
cbit - after this # of iterations, output is printed
verb - if output should be produced
rdelta - stopping condition for reconstruction error
history - MVHistory() to be filled with data of individual iterations
"""
function fit!(ae::AutoencoderWithMemory, X, L; iterations=1000, cbit = 200, verb = true, rdelta = Inf, history = nothing)

	# optimizer
	opt = ADAM(params(Chain(ae.encoder, ae.decoder)))

	# training
	for i in 1:iterations
		# sample from data
		x = X[:, sample(1:size(X,2), L, replace = false)]

		# gradient computation and update
		l = learnRepresentation(ae, x)
		Flux.Tracker.back!(l)
		opt()

		# callback
		if verb && (i % cbit == 0)
			evalReconstructionError(ae, x)
		end

		# save actual iteration data
		if history != nothing
			track!(ae, history, x)
		end

		# if stopping condition is present
		if rdelta < Inf
			re = Flux.Tracker.data(l)[1]
			if re < rdelta
				if verb
					println("Training ended prematurely after $i iterations,\n",
						"reconstruction error $re < $rdelta")
				end
				break
			end
		end
	end
end

function fitAnomaly!(ae::AutoencoderWithMemory, anomaly, label = 1, γ = 0.5)
	opt = ADAM(params(ae.encoder))
	l = learnAnomaly(ae, anomaly, label, γ)
	Flux.Tracker.back!(l)
	opt()
end

"""
	track!(ae, history, X)

Save current progress.
"""
function track!(ae::AutoencoderWithMemory, history::MVHistory, X)
	push!(history, :loss, Flux.Tracker.data(reconstructionError(ae,X)))
end

#################
### ae output ###
#################

"""
	anomalyscore(ae, X)

Compute anomaly score for X.
"""
function anomalyscore(ae::AutoencoderWithMemory, X::Array{Float, N} where N)
	(values, probabilities) = query(ae.memory, X)
	probabilities[values .== 0] = 0
	return probabilities
end

"""
	classify(ae, x, threshold)

Classify an instance x using reconstruction error and threshold.
"""
classify(ae::AutoencoderWithMemory, X, threshold) = Int.(anomalyscore(ae, X) .> threshold)

"""
	getthreshold(ae, x, contamination, [Beta])

Compute threshold for AE classification based on known contamination level.
"""
function getthreshold(ae::AutoencoderWithMemory, x, contamination; Beta = 1.0)
	N = size(x, 2)
	Beta = Float(Beta)
	# get reconstruction errors
	ascore = anomalyscore(ae, x)
	# sort it
	ascore = sort(ascore)
	aN = Int(ceil(N*contamination)) # number of contaminated samples
	# get the threshold - could this be done more efficiently?
	(aN > 0)? (return Beta*ascore[end-aN] + (1-Beta)*ascore[end-aN+1]) : (return ascore[end])
end

#############################################################################
### An SK-learn like model based on AE with the same methods and some new. ###
#############################################################################

"""
Struct to be used as scikitlearn-like model with fit and predict methods.
"""
mutable struct AutoencoderWithMemoryModel <: genmodel
	ae::AutoencoderWithMemory
	L::Int
	threshold::Real
	contamination::Real
	iterations::Int
	cbit::Real
	verbfit::Bool
	γ::Float
	rdelta::Real
	Beta::Float
	history
end

"""
	AEmodel(esize, dsize, L, threshold, contamination, iteration, cbit,
	[activation, rdelta, Beta, tracked])

Initialize an autoencoder model with given parameters.

esize - encoder architecture
dsize - decoder architecture
L - batchsize
threshold - anomaly score threshold for classification, is set automatically using contamination during fit
contamination - percentage of anomalous samples in all data for automatic threshold computation
iterations - number of training iterations
cbit - current training progress is printed every cbit iterations
verbfit - is progress printed?
activation [Flux.relu] - activation function
rdelta [Inf] - training stops if reconstruction error is smaller than rdelta
Beta [1.0] - how tight around normal data is the automatically computed threshold
tracked [false] - is training progress (losses) stored?
"""
function AutoencoderWithMemoryModel(esize::Array{Int64,1}, dsize::Array{Int64,1}, L::Int, threshold::Real, contamination::Real, iterations::Int,
	cbit::Real, verbfit::Bool, memorySize::Integer, keySize::Integer, k::Integer; labelCount = 2, α = 0.1, γ = 0.5, rdelta = Inf, Beta = 1.0,
	tracked = false, activation = Flux.relu, layer = ResDense)

	# construct the AE object
	ae = AutoencoderWithMemory(esize, dsize, memorySize, keySize, k, labelCount, α, activation, layer)
	(tracked)? history = MVHistory() : history = nothing
	model = AutoencoderWithMemoryModel(ae, L, threshold, contamination, iterations, cbit, verbfit, γ, rdelta, Beta, history)
	return model
end

# reimplement some methods of AE
loss(model::AutoencoderWithMemoryModel, X) = reconstructionError(model.ae, X)
evalloss(model::AutoencoderWithMemoryModel, X) = evalReconstructionError(model.ae, X)
anomalyscore(model::AutoencoderWithMemoryModel, X) = anomalyscore(model.ae, X)
classify(model::AutoencoderWithMemoryModel, x) = classify(model.ae, x, model.threshold)
getthreshold(model::AutoencoderWithMemoryModel, x) = getthreshold(model.ae, x, model.contamination, Beta = model.Beta)

"""
	setthreshold!(model::AEmodel, X)

Set model classification threshold based ratior of labels in Y.
"""
function setthreshold!(model::AutoencoderWithMemoryModel, X)
	model.threshold = getthreshold(model, X)
end

"""
	fit!(model::AEmodel, X)

Fit the AE model, instances are columns of X, X are normal samples!!!.
"""
function fit!(model::AutoencoderWithMemoryModel, X)
	# train
	fit!(model.ae, X, model.L, iterations = model.iterations, cbit = model.cbit, verb = model.verbfit, rdelta = model.rdelta, history = model.history)
end

"""
	predict(model::AEmodel, X)

Based on known , label = 1, γ = 0.5contamination level, compute threshold and classify instances in X.
"""
function predict(model::AutoencoderWithMemoryModel, X)
	return classify(model, X)
end

function fitAnomaly!(model::AutoencoderWithMemoryModel, data)
	fitAnomaly!(model.ae, data, label, ae.γ)
end
