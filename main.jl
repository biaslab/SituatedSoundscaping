using Revise
using SituatedSoundscaping

# settings 
nr_mixtures = 10
nr_files_speech = 100
nr_files_noise = 100
nr_iterations_em = 10
nr_iterations_gs = 10
nr_iterations_adjust = 10
observation_noise_precision = 1e5

# prepare data
data_speech = prepare_data("data/train_speech_raw", "data/train_speech_processed")
data_noise = prepare_data("data/train_noise_raw", "data/train_noise_processed")
nr_files_speech = minimum([nr_files_speech, length(data_speech)])
nr_files_noise = minimum([nr_files_noise, length(data_noise)])
data_speech = data_speech[1:nr_files_speech]
data_noise = data_noise[1:nr_files_noise]
recording_speech = read_recording("data/recorded_speech_processed/recording_speech.h5", duration=3)
recording_noise = read_recording("data/recorded_noise_processed/recording_noise.h5", duration=3)

# train Kmeans models
centers_speech, πk1_speech = train_kmeans("models/Kmeans/speech", data_speech, nr_mixtures)
centers_noise, πk1_noise = train_kmeans("models/Kmeans/noise", data_noise, nr_mixtures)

# train EM models
means_speech, covs_speech, πk2_speech = train_em("models/EM/speech", data_speech, centers_speech, πk1_speech; nr_iterations=nr_iterations_em)
means_noise, covs_noise, πk2_noise = train_em("models/EM/noise", data_noise, centers_noise, πk1_noise; nr_iterations=nr_iterations_em)

# train GS models
q_μ_speech, q_γ_speech, q_a_speech = train_gs("models/GS/speech", data_speech, means_speech, covs_speech, πk2_speech; nr_iterations=nr_iterations_gs, observation_noise=observation_noise_precision);
q_μ_noise, q_γ_noise, q_a_noise = train_gs("models/GS/noise", data_noise, means_noise, covs_noise, πk2_noise; nr_iterations=nr_iterations_gs, observation_noise=observation_noise_precision);

# adjust model on recording
p_full_speech, q_full_speech = update_model("models/adjusted/speech", recording_speech, (q_a_speech, q_μ_speech, q_γ_speech), nr_files_speech; nr_iterations=nr_iterations_adjust, observation_noise=observation_noise_precision)
p_full_noise, q_full_noise = update_model("models/adjusted/noise", recording_noise, (q_a_noise, q_μ_noise, q_γ_noise), nr_files_noise; nr_iterations=nr_iterations_adjust, observation_noise=observation_noise_precision)

# perform Bayesian model reduction (2 approaches)
p_red1_speech, q_red1_speech, Δp1_speech = model_reduction_all(p_full_speech, q_full_speech)
p_red2_speech, q_red2_speech, Δp2_speech = model_reduction_steps(p_full_speech, q_full_speech)
p_red1_noise, q_red1_noise, Δp1_noise = model_reduction_all(p_full_noise, q_full_noise)
p_red2_noise, q_red2_noise, Δp2_noise = model_reduction_steps(p_full_noise, q_full_noise)

# simplify models
q_μ_speech, q_γ_speech, q_a_speech = simplify_model(q_μ_speech, q_γ_speech, p_red1_speech)
q_μ_noise, q_γ_noise, q_a_noise = simplify_model(q_μ_noise, q_γ_noise, p_red1_noise)
