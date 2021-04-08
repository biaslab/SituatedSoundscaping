using Revise
using SituatedSoundscaping

# settings 
nr_mixtures = 10
nr_files = 10
nr_iterations_em = 10
nr_iterations_gs = 10
nr_iterations_adjust = 10

# prepare data
data = prepare_data("data/train_speech_raw", "data/train_speech_processed")
data = data[1:nr_files]
recording = read_recording("data/recorded_speech_processed/recording_speech.h5", duration=3)

# train Kmeans model
centers, πk1 = train_kmeans("models/Kmeans/speech", data, nr_mixtures)

# train EM model
means, covs, πk2 = train_em("models/EM/speech", data, centers, πk1; nr_iterations=nr_iterations_em)

# train GS model
q_μ, q_γ, q_a = train_gs("models/GS/speech", data, means, covs, πk2; nr_iterations=nr_iterations_gs);

# adjust model on recording
p_full, q_full = update_model("models/adjusted/speech", recording, (q_a, q_μ, q_γ), nr_files; nr_iterations=nr_iterations_adjust)

# perform Bayesian model reduction
p_red1, q_red1, Δp1 = model_reduction_all(p_full, q_full)
p_red2, q_red2, Δp2 = model_reduction_steps(p_full, q_full)