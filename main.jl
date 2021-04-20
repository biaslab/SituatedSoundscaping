using Revise
using SituatedSoundscaping

# settings 
nr_mixtures_speech = 100
nr_mixtures_noise = 5
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

# perform source separation
mixed_signal, speech_signal, noise_signal = create_mixture_signal("data/recorded_speech_raw/recording_speech.flac", "data/recorded_noise_raw/recording_noise.wav", duration_adapt=3, duration_test=1)
speech_out, G = separate_sources(mixed_signal, q_μ_speech, q_γ_speech, q_a_speech, q_μ_noise, q_γ_noise, q_a_noise; observation_noise_precision=observation_noise_precision)

# calculate metrics
SNRo = SNR(speech_signal, speech_out)
pesqw = PESQ(16000, speech_signal, speech_out, "wb")
pesqn = PESQ(16000, speech_signal, speech_out, "nb")
stoi = STOI(speech_signal, speech_out, 16000, extended=false)

# find optimal values
output, X, S, N, G_wiener = separate_sources_wiener(mixed_signal, speech_signal, noise_signal)


using PyPlot
plt.figure()
plt.imshow(hcat(mean.(q_μ_noise)...), aspect="auto", origin="lower", cmap="jet")
plt.colorbar()
plt.gcf()

tmp = hcat(mean.(q_μ_noise)...) -  1 ./ hcat(mean.(q_γ_noise)...)/2
plt.figure()
plt.imshow(tmp, aspect="auto", origin="lower", cmap="jet")
plt.colorbar()
plt.gcf()


plt.figure()
plt.imshow(log.(abs2.(G))', origin="lower", cmap="jet", aspect="auto")
plt.colorbar()
plt.gcf()

plt.figure()
plt.imshow(log.(abs2.(G_wiener))', origin="lower", cmap="jet", aspect="auto")
plt.colorbar()
plt.gcf()






_, ax = plt.subplots(ncols=3, figsize=(15,10))
ax[1].imshow(log.(abs2.(X')), origin="lower", aspect="auto", cmap="jet")
ax[2].imshow(log.(abs2.(S')), origin="lower", aspect="auto", cmap="jet")
ax[3].imshow(log.(abs2.(N')), origin="lower", aspect="auto", cmap="jet")
plt.gcf()

plt.figure()
plt.plot(speech_out)
plt.grid()
plt.gcf()






using WAV
wavwrite(speech_out, "x_separated_speech.wav", Fs=16000)
wavwrite(speech_signal, "x_true_speech.wav", Fs=16000)
wavwrite(mixed_signal, "x_mixed.wav", Fs=16000)
wavwrite(output, "x_wiener.wav", Fs=16000)
using PyPlot, DSP

plt.figure()
plt.plot(mixed_signal .+ 5)
plt.plot(speech_out .- 5)
plt.plot(speech_signal .- 15)
plt.grid()
plt.gcf()

plt.figure()
plt.imshow(log.(abs2.(G))', aspect="auto", origin="lower")
plt.gcf()

_, ax = plt.subplots(ncols=3, figsize=(15,10))
ax[1].imshow(log.(abs2.(stft(mixed_signal, 64))), aspect="auto", origin="lower", cmap="jet")
ax[2].imshow(log.(abs2.(stft(speech_signal, 64))), aspect="auto", origin="lower", cmap="jet")
ax[3].imshow(log.(abs2.(stft(speech_out, 64))), aspect="auto", origin="lower", cmap="jet")
plt.gcf()

_, ax = plt.subplots(ncols=3, figsize=(15,10))
ax[1].imshow(log.(abs2.(stft(mixed_signal, 64))), aspect="auto", origin="lower", cmap="jet")
ax[2].imshow(log.(abs2.(stft(noise_signal, 64))), aspect="auto", origin="lower", cmap="jet")
ax[3].imshow(log.(abs2.(stft(noise_out, 64))), aspect="auto", origin="lower", cmap="jet")
plt.gcf()

_, ax = plt.subplots(ncols=3, figsize=(15,10))
ax[1].imshow(centers_speech, origin="lower", aspect="auto", cmap="jet")
ax[2].imshow(means_speech, origin="lower", aspect="auto", cmap="jet")
ax[3].imshow(hcat(mean.(q_μ_speech)...), origin="lower", aspect="auto", cmap="jet")
plt.gcf()

_, ax = plt.subplots(ncols=3, figsize=(15,10))
ax[1].imshow(centers_noise, origin="lower", aspect="auto", cmap="jet")
ax[2].imshow(means_noise, origin="lower", aspect="auto", cmap="jet")
ax[3].imshow(hcat(mean.(q_μ_noise)...), origin="lower", aspect="auto", cmap="jet")
plt.gcf()





