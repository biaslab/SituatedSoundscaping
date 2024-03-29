using Revise
using SituatedSoundscaping

# settings
nr_mixtures_speech = [10, 15, 20, 25, 50, 100]
nr_mixtures_noise = [1, 2, 3, 4, 5, 10]
nr_files_speech = 1000
nr_iterations_em = 10
observation_noise_precision = 10.0# 4e0
power_noise = [-10, -5, 0, 5, 10]
overwrite = false

# perform experiments
data_speech = prepare_data("data/train_speech_raw", "data/train_speech_processed32"; block_length=32)
nr_files_speech = minimum([nr_files_speech, length(data_speech)])
data_speech = data_speech[1:nr_files_speech]

for (ndB, nmix_speech, nmix_noise) in Iterators.product(power_noise, nr_mixtures_speech, nr_mixtures_noise)

    # create identifier
    nr_freqs = 32 ÷ 2 + 1
    id = string("_freq=", nr_freqs, "_mixs=", nmix_speech, "_mixn=", nmix_noise, "_power=", ndB)

    # skip if already done
    if !overwrite 
        if isfile("exports/algonquin_paper/metrics"*id*".txt")
            continue
        end
    end

    # prepare data sets
    prepare_data("data/recorded_noise_raw", "data/recorded_noise_processed32/"*string(ndB); block_length=32, power_dB=ndB)
    recording_noise = read_recording("data/recorded_noise_processed32/"*string(ndB)*"/0000000001.h5", duration=3)
    mixed_signal, speech_signal, noise_signal = create_mixture_signal("data/recorded_speech_raw/recording_speech.flac", "data/recorded_noise_raw/recording_noise.wav", duration_adapt=3.0, duration_test=10.0, power_noise_dB=ndB)

    # train model speech
    centers_speech, πk1_speech = train_kmeans("models/Kmeans/speech", data_speech, nmix_speech)
    means_speech, covs_speech, πk2_speech = train_em("models/EM/speech", data_speech, centers_speech, πk1_speech; nr_iterations=nr_iterations_em)

    # train noise model on recording
    centers_noise, πk1_noise = train_kmeans("models/Kmeans/noise", log.(abs2.(recording_noise)), nmix_noise; power_dB=ndB)
    means_noise, covs_noise, πk2_noise = train_em("models/EM/noise", log.(abs2.(recording_noise)), centers_noise, πk1_noise; nr_iterations=nr_iterations_em, power_dB=ndB)

    # perform source separation
    speech_sep = separate_sources_algonquin_paper("exports/algonquin_paper", mixed_signal, means_speech, covs_speech, πk2_speech, means_noise, covs_noise, πk2_noise; observation_noise_precision=observation_noise_precision, block_length=32, power_dB=ndB, save_results=true)

    # evaluate metrics
    evaluate_metrics("exports/algonquin_paper/metrics"*id*".txt", speech_sep, mixed_signal, speech_signal)

end