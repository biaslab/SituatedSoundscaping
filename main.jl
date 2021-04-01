using Revise
using SituatedSoundscaping

# settings 
nr_mixtures = 100
nr_files = 10
nr_iterations_em = 10
nr_iterations_gs = 10

# prepare data
data = prepare_data("data/train_speech_raw", "data/train_speech_processed")
data = data[1:nr_files]

# train Kmeans model
centers, πk1 = train_kmeans("models/Kmeans/speech", data, nr_mixtures)

# train EM model
means, covs, πk2 = train_em("models/EM/speech", data, centers, πk1; nr_iterations=nr_iterations_em)

# train GS model
q_μ, q_γ, q_a = train_gs("models/GS/speech", data, means, covs, πk2; nr_iterations=nr_iterations_gs);

