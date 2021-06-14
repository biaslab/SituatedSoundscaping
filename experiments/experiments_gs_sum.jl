using Revise
using SituatedSoundscaping

# settings
nr_mixtures_speech = [10, 15, 20, 25, 50]
nr_mixtures_noise = [1, 2, 3, 4, 5, 10]
nr_files_speech = 1000
nr_iterations_em = 10
nr_iterations_gs = 10
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
        if isfile("exports/gs_sum/metrics"*id*".txt")
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
    q_μ_speech, q_γ_speech, q_a_speech = train_gs("models/GS/speech", data_speech, means_speech, covs_speech, πk2_speech; nr_iterations=nr_iterations_gs);

    # train noise model on recording
    centers_noise, πk1_noise = train_kmeans("models/Kmeans/noise", log.(abs2.(recording_noise)), nmix_noise; power_dB=ndB)
    means_noise, covs_noise, πk2_noise = train_em("models/EM/noise", log.(abs2.(recording_noise)), centers_noise, πk1_noise; nr_iterations=nr_iterations_em, power_dB=ndB)
    q_μ_noise, q_γ_noise, q_a_noise = train_gs("models/GS/noise", recording_noise, means_noise, covs_noise, πk2_noise; nr_iterations=nr_iterations_gs, power_dB=ndB);

    # perform source separation
    speech_sep = separate_sources_gs_sum("exports/gs_sum", mixed_signal, q_μ_speech, q_γ_speech, q_a_speech, q_μ_noise, q_γ_noise, q_a_noise; block_length=32, power_dB=ndB, save_results=true, nr_iterations=nr_iterations_gs)

    # evaluate metrics
    evaluate_metrics("exports/gs_sum/metrics"*id*".txt", speech_sep, mixed_signal, speech_signal)

end