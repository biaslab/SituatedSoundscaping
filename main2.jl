using Revise
using SituatedSoundscaping

# settings 
nr_mixtures_speech = 25
nr_mixtures_noise = 20
nr_files_speech = 1000
nr_files_noise = 1000
nr_iterations_gs = 10
nr_iterations_em = 10
nr_iterations_adjust = 10
observation_noise_precision = 1e3

# prepare data
data_speech = prepare_data("data/train_speech_raw", "data/train_speech_processed")
data_noise = prepare_data("data/train_noise_raw", "data/train_noise_processed")
prepare_data("data/recorded_speech_raw", "data/recorded_speech_processed")
prepare_data("data/recorded_noise_raw", "data/recorded_noise_processed")
nr_files_speech = minimum([nr_files_speech, length(data_speech)])
nr_files_noise = minimum([nr_files_noise, length(data_noise)])
data_speech = data_speech[1:nr_files_speech]
data_noise = data_noise[1:nr_files_noise]
recording_speech = read_recording("data/recorded_speech_processed/recording_speech.h5", duration=3)
recording_noise = read_recording("data/recorded_noise_processed/recording_noise.h5", duration=3)

# train Kmeans models
centers_speech, πk1_speech = train_kmeans("models/Kmeans/speech", data_speech, nr_mixtures_speech)
centers_noise, πk1_noise = train_kmeans("models/Kmeans/noise", data_noise, nr_mixtures_noise)

# train EM models
means_speech, covs_speech, πk2_speech = train_em("models/EM/speech", data_speech, centers_speech, πk1_speech; nr_iterations=nr_iterations_em)
means_noise, covs_noise, πk2_noise = train_em("models/EM/noise", data_noise, centers_noise, πk1_noise; nr_iterations=nr_iterations_em)

# train GS models
q_μ_speech, q_γ_speech, q_a_speech = train_gs("models/GS/speech", data_speech, means_speech, covs_speech, πk2_speech; nr_iterations=nr_iterations_gs, observation_noise_precision=observation_noise_precision);
q_μ_noise, q_γ_noise, q_a_noise = train_gs("models/GS/noise", data_noise, means_noise, covs_noise, πk2_noise; nr_iterations=nr_iterations_gs, observation_noise_precision=observation_noise_precision);

# q_full_speech = q_a_speech
using Distributions
p_full_speech = Dirichlet(ones(nr_mixtures_speech))
q_full_speech = Dirichlet(q_a_speech.a)
# Dcategorical(exp.(logmean(p_full_speech)) ./ sum(exp.(logmean(p_full_speech))))

using PyPlot
_, ax = plt.subplots(ncols=2)
ax[1].imshow(hcat(mean.(q_μ_speech)...), aspect="auto", origin="lower", cmap="jet")
ax[2].imshow(means_speech, aspect="auto", origin="lower", cmap="jet")
plt.gcf()
_, ax = plt.subplots(ncols=2)
ax[1].imshow(hcat(mean.(q_γ_speech)...), aspect="auto", origin="lower", cmap="jet")
ax[2].imshow(1 ./ covs_speech, aspect="auto", origin="lower", cmap="jet")
plt.gcf()
_, ax = plt.subplots(ncols=2)
ax[1].bar(collect(1:nr_mixtures_speech), q_a_speech.a)
ax[2].bar(collect(1:nr_mixtures_speech), πk2_speech)
plt.gcf()

# perform Bayesian model reduction (2 approaches)
model_reduction_info(p_full_speech, q_full_speech)

# simplify models
q_μ_speech, q_γ_speech, q_a_speech = simplify_model(q_μ_speech, q_γ_speech, p_red1_speech)
q_μ_noise, q_γ_noise, q_a_noise = simplify_model(q_μ_noise, q_γ_noise, p_red1_noise)

# perform source separation
mixed_signal, speech_signal, noise_signal = create_mixture_signal("data/recorded_speech_raw/recording_speech.flac", "data/recorded_noise_raw/recording_noise.wav", duration_adapt=3.0, duration_test=5.0)
speech_out, G = separate_sources(mixed_signal, q_μ_speech, q_γ_speech, q_a_speech, q_μ_noise, q_γ_noise, q_a_noise; observation_noise_precision=observation_noise_precision)

# calculate metrics
SNRo = SNR(speech_signal, speech_out)
pesqw = PESQ(16000, speech_signal, speech_out, "wb")
pesqn = PESQ(16000, speech_signal, speech_out, "nb")
stoi = STOI(speech_signal, speech_out, 16000, extended=false)

# find optimal values
output, X, S, N, G_wiener = separate_sources_wiener(mixed_signal, speech_signal, noise_signal)

