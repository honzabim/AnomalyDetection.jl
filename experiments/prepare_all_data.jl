# creates folders from all loda datasets given the other makeset settings -
# alpha, difficulty, frequency, variation and seed
# either change the settings in the file or call the script with arguments
# julia prepare_all_data.jl alpha difficulty frequency variation seed
# e. g. julia prepare_all_data.jl 0.7 easy 0.02 low 12345

push!(LOAD_PATH, ".")
using Experiments

# settings
nargs = size(ARGS, 1)
# ratio of training to all data
(nargs > 0)? alpha = parse(Float64, ARGS[1]) : alpha = 0.8 
# easy/medium/hard/very_hard problem based on similarity of anomalous measurements to normal
(nargs > 1)? difficulty = ARGS[2] : difficulty = "easy" 
# ratio of anomalous to normal data\n
(nargs > 2)? frequency = parse(Float64, ARGS[3]) : frequency = 0.02 
# low/high - should anomalies be clustered or not
(nargs > 3)? variation = ARGS[4] : variation = "low"
# number of repetitions
(nargs > 4)? repetition = parse(Int64, ARGS[5]) : repetition = 0
# random seed 
(nargs > 5)? seed = parse(Int64, ARGS[6]) : seed = false 

files = readdir(Experiments.loda_path)
for dataset_name in files
	try
		Experiments.prepare_data(dataset_name, alpha, difficulty, frequency, variation, seed,
			repetition = repetition, verb = true)
	catch
		println("$(dataset_name) export unsuccesful!")
	end
end
