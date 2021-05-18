using Revise
using SituatedSoundscaping
using BenchmarkTools

# settings 
nr_mixtures_speech = 100
nr_mixtures_noise = 10
nr_files_speech = 1000
nr_files_noise = 100
nr_iterations_em = 10
observation_noise_precision = 10.0# 4e0

# prepare data
data_speech = prepare_data("data/train_speech_raw", "data/train_speech_processed")
data_noise = prepare_data("data/train_noise_raw", "data/train_noise_processed")
data_speech = prepare_data("data/train_speech_raw", "data/train_speech_processed32"; block_length=32)
data_noise = prepare_data("data/train_noise_raw", "data/train_noise_processed32"; block_length=32)
prepare_data("data/recorded_speech_raw", "data/recorded_speech_processed")
prepare_data("data/recorded_noise_raw", "data/recorded_noise_processed")
prepare_data("data/recorded_noise_raw", "data/recorded_noise_processed32", block_length=32)
nr_files_speech = minimum([nr_files_speech, length(data_speech)])
nr_files_noise = minimum([nr_files_noise, length(data_noise)])
data_speech = data_speech[1:nr_files_speech]
data_noise = data_noise[1:nr_files_noise]
recording_speech = read_recording("data/recorded_speech_processed/recording_speech.h5", duration=3)
recording_noise = read_recording("data/recorded_noise_processed/recording_noise.h5", duration=3)

# train Kmeans models
centers_speech, πk1_speech = train_kmeans("models/Kmeans/speech32", data_speech, nr_mixtures_speech)
centers_noise, πk1_noise = train_kmeans("models/Kmeans/noise32", data_noise, nr_mixtures_noise)

# train EM models
means_speech, covs_speech, πk2_speech = train_em("models/EM/speech32", data_speech, centers_speech, πk1_speech; nr_iterations=nr_iterations_em)
means_noise, covs_noise, πk2_noise = train_em("models/EM/noise32", data_noise, centers_noise, πk1_noise; nr_iterations=nr_iterations_em)

# train new noise model on recording
#data_noise = SituatedSoundscaping.Data(["data/recorded_noise_processed/recording_noise.h5"], Float64)
data_noise = log.(abs2.(read_recording("data/recorded_noise_processed32/recording_noise.h5", duration=3, block_length=32)))
centers_noise, πk1_noise = train_kmeans("models/Kmeans/noiserec32", data_noise, nr_mixtures_noise)
means_noise, covs_noise, πk2_noise = train_em("models/EM/noiserec32", data_noise, centers_noise, πk1_noise; nr_iterations=nr_iterations_em)

# perform source separation
mixed_signal, speech_signal, noise_signal = create_mixture_signal("data/recorded_speech_raw/recording_speech.flac", "data/recorded_noise_raw/recording_noise.wav", duration_adapt=3.0, duration_test=5.0)
speech_out, G = separate_sources_algonquin(mixed_signal, means_speech, covs_speech, πk2_speech, means_noise, covs_noise, πk2_noise; observation_noise_precision=observation_noise_precision, block_length=32)
output, X, S, N, G_wiener = separate_sources_wiener(mixed_signal, speech_signal, noise_signal; block_length=32)

filterbank = SituatedSoundscaping.warped_filter_bank(block_duration_s=32/16000, nr_bands=Int(32/2)+1)
block_length = 32
nr_blocks = Int(length(speech_signal)/block_length)
nr_freqs = block_length ÷ 2 + 1
output_speech = zeros(size(speech_signal))
SituatedSoundscaping.update_fir_weights!(filterbank, collect(17:-1:1)/17)
for k in 1:nr_blocks

    # feed signals into filterbanks
    SituatedSoundscaping.run!(filterbank, speech_signal[1+(k-1)*block_length:k*block_length])

    # read from filter
    output_speech[1+(k-1)*block_length:k*block_length] = read(filterbank)

end
# wavwrite(output_speech, "x_warped.wav", Fs=16000)

SNRo = SNR(output_speech, speech_out)
pesqw = PESQ(16000, output_speech, speech_out, "wb")
pesqn = PESQ(16000, output_speech, speech_out, "nb")
stoi = STOI(output_speech, speech_out, 16000, extended=false)

# plt.figure()
# plt.plot(output)
# plt.plot(speech_signal)
# plt.plot(output_speech)
# plt.xlim(0, 100)
# plt.gcf()


using PyPlot

plt.figure()
plt.imshow(log.(abs2.(G))', origin="lower", cmap="jet", aspect="auto")
plt.colorbar()
plt.clim(-35.0)
plt.gcf()

plt.figure()
plt.imshow(log.(abs2.(G_wiener))', origin="lower", cmap="jet", aspect="auto")
plt.colorbar()
plt.gcf()

plt.figure()
plt.plot(mixed_signal .+ 5)
plt.plot(speech_out .- 5)
plt.plot(speech_signal .- 15)
plt.grid()
plt.gcf()

_, ax = plt.subplots(ncols=3, figsize=(15,10))
ax[1].imshow(log.(abs2.(X')), origin="lower", aspect="auto", cmap="jet")
ax[2].imshow(log.(abs2.(S')), origin="lower", aspect="auto", cmap="jet")
ax[3].imshow(log.(abs2.(N')), origin="lower", aspect="auto", cmap="jet")
plt.gcf()

1+1


using WAV
wavwrite(normalize_range(speech_out), "x_separated_speech_algonquin.wav", Fs=16000)
wavwrite(normalize_range(speech_signal), "x_true_speech.wav", Fs=16000)
wavwrite(normalize_range(mixed_signal), "x_mixed.wav", Fs=16000)
wavwrite(normalize_range(output), "x_wiener.wav", Fs=16000)
wavwrite(normalize_range(0.3*mixed_signal + 0.7*speech_out), "x_weighted.wav", Fs=16000)

_, ax = plt.subplots(ncols=2)
ax[1].imshow(means_speech', aspect="auto", origin="lower", cmap="jet")
ax[2].imshow(means_noise', aspect="auto", origin="lower", cmap="jet")
plt.gcf()