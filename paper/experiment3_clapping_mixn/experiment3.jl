using SituatedSoundscaping


# experiments algonquin_vb
begin 
    # settings
    nr_mixtures_speech = [25]
    nr_mixtures_noise = [1,2,3,4,5]
    nr_files_speech = 1000
    nr_iterations_em = 10
    nr_iterations_vb = 10
    nr_iterations_vmp = 10
    observation_noise_precision = 10.0
    power_noise = [0]
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
            if isfile("paper/experiment1_clapping_SNR/exports/algonquin_vb/metrics"*id*".txt")
                continue
            end
        end

        # prepare data sets
        prepare_data("data/recorded_noise_raw", "data/recorded_noise_processed32/"*string(ndB); block_length=32, power_dB=ndB)
        recording_noise = read_recording("data/recorded_noise_processed32/"*string(ndB)*"/0000000001.h5", duration=3)
        mixed_signal, speech_signal, noise_signal = create_mixture_signal("data/recorded_speech_raw/recording_speech.flac", "data/recorded_noise_raw/recording_noise.wav", duration_adapt=3.0, duration_test=10.0, power_noise_dB=ndB)

        # train speech mode
        centers_speech, πk1_speech = train_kmeans("paper/experiment1_clapping_SNR/exports/models/Kmeans_speech", data_speech, nmix_speech)
        means_speech, covs_speech, πk2_speech = train_em("paper/experiment1_clapping_SNR/exports/models/EM_speech", data_speech, centers_speech, πk1_speech; nr_iterations=nr_iterations_em)
        q_μ_speech, q_γ_speech, q_a_speech = train_vb("paper/experiment1_clapping_SNR/exports/models/VB_speech", data_speech, means_speech, covs_speech, πk2_speech; nr_iterations=nr_iterations_vb);

        # train noise model on recording
        centers_noise, πk1_noise = train_kmeans("paper/experiment1_clapping_SNR/exports/models/Kmeans_noise", log.(abs2.(recording_noise)), nmix_noise; power_dB=ndB)
        means_noise, covs_noise, πk2_noise = train_em("paper/experiment1_clapping_SNR/exports/models/EM_noise", log.(abs2.(recording_noise)), centers_noise, πk1_noise; nr_iterations=nr_iterations_em, power_dB=ndB)
        q_μ_noise, q_γ_noise, q_a_noise = train_vb("paper/experiment1_clapping_SNR/exports/models/VB_noise", log.(abs2.(recording_noise)), means_noise, covs_noise, πk2_noise; nr_iterations=nr_iterations_vb, power_dB=ndB);

        # perform source separation
        speech_sep = separate_sources_algonquin_vb("paper/experiment1_clapping_SNR/exports/algonquin_vb", mixed_signal, q_μ_speech, q_γ_speech, q_a_speech, q_μ_noise, q_γ_noise, q_a_noise; observation_noise_precision=observation_noise_precision, block_length=32, power_dB=ndB, save_results=true, nr_iterations=nr_iterations_vmp)

        # evaluate metrics
        evaluate_metrics("paper/experiment1_clapping_SNR/exports/algonquin_vb/metrics"*id*".txt", speech_sep, mixed_signal, speech_signal)
        
    end
end
# END:  experiments algonquin_vb


# experiments gs_sum
begin
        
    # settings
    nr_mixtures_speech = [25]
    nr_mixtures_noise = [1, 2, 3, 4, 5]
    nr_files_speech = 1000
    nr_iterations_em = 10
    nr_iterations_gs = 10
    power_noise = [0]
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
            if isfile("paper/experiment1_clapping_SNR/exports/gs_sum/metrics"*id*".txt")
                continue
            end
        end

        # prepare data sets
        prepare_data("data/recorded_noise_raw", "data/recorded_noise_processed32/"*string(ndB); block_length=32, power_dB=ndB)
        recording_noise = read_recording("data/recorded_noise_processed32/"*string(ndB)*"/0000000001.h5", duration=3)
        mixed_signal, speech_signal, noise_signal = create_mixture_signal("data/recorded_speech_raw/recording_speech.flac", "data/recorded_noise_raw/recording_noise.wav", duration_adapt=3.0, duration_test=10.0, power_noise_dB=ndB)

        # train model speech
        centers_speech, πk1_speech = train_kmeans("paper/experiment1_clapping_SNR/exports/models/Kmeans_speech", data_speech, nmix_speech)
        means_speech, covs_speech, πk2_speech = train_em("paper/experiment1_clapping_SNR/exports/models/EM_speech", data_speech, centers_speech, πk1_speech; nr_iterations=nr_iterations_em)
        q_μ_speech, q_γ_speech, q_a_speech = train_gs("paper/experiment1_clapping_SNR/exports/models/GS_speech", data_speech, means_speech, covs_speech, πk2_speech; nr_iterations=nr_iterations_gs);

        # train noise model on recording
        centers_noise, πk1_noise = train_kmeans("paper/experiment1_clapping_SNR/exports/models/Kmeans_noise", log.(abs2.(recording_noise)), nmix_noise; power_dB=ndB)
        means_noise, covs_noise, πk2_noise = train_em("paper/experiment1_clapping_SNR/exports/models/EM_noise", log.(abs2.(recording_noise)), centers_noise, πk1_noise; nr_iterations=nr_iterations_em, power_dB=ndB)
        q_μ_noise, q_γ_noise, q_a_noise = train_gs("paper/experiment1_clapping_SNR/exports/models/GS_noise", recording_noise, means_noise, covs_noise, πk2_noise; nr_iterations=nr_iterations_gs, power_dB=ndB);

        # perform source separation
        speech_sep = separate_sources_gs_sum("paper/experiment1_clapping_SNR/exports/gs_sum", mixed_signal, q_μ_speech, q_γ_speech, q_a_speech, q_μ_noise, q_γ_noise, q_a_noise; block_length=32, power_dB=ndB, save_results=true, nr_iterations=nr_iterations_gs)

        # evaluate metrics
        evaluate_metrics("paper/experiment1_clapping_SNR/exports/gs_sum/metrics"*id*".txt", speech_sep, mixed_signal, speech_signal)

    end
end
# END: experiments gs_sum


# experiments Wiener
begin
    
    # settings
    power_noise = [-10, -5, 0, 5, 10]

    # perform experiments
    for ndB in power_noise

        # prepare id
        id = string("_power=", ndB)

        # prepare data
        mixed_signal, speech_signal, noise_signal = create_mixture_signal("data/recorded_speech_raw/recording_speech.flac", "data/recorded_noise_raw/recording_noise.wav", duration_adapt=3.0, duration_test=10.0, power_noise_dB=ndB)

        # separate sources
        speech_sep, _, _, _, _ = separate_sources_wiener("paper/experiment1_clapping_SNR/exports/wiener", mixed_signal, speech_signal, noise_signal; block_length=32, power_dB=ndB, save_results=true)

        # evaluate metrics
        evaluate_metrics("paper/experiment1_clapping_SNR/exports/wiener/metrics"*id*".txt", speech_sep, mixed_signal, speech_signal)

    end
end