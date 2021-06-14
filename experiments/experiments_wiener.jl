using Revise
using SituatedSoundscaping

# settings
power_noise = [-10, -5, 0, 5, 10]

# perform experiments
for ndB in power_noise

    # prepare id
    id = string("_power=", ndB)

    # prepare data
    mixed_signal, speech_signal, noise_signal = create_mixture_signal("data/recorded_speech_raw/recording_speech.flac", "data/recorded_noise_raw/recording_noise.wav", duration_adapt=3.0, duration_test=10.0, power_noise_dB=ndB)

    # separate sources
    speech_sep, _, _, _, _ = separate_sources_wiener("exports/wiener", mixed_signal, speech_signal, noise_signal; block_length=32, power_dB=ndB, save_results=true)

    # evaluate metrics
    evaluate_metrics("exports/wiener/metrics"*id*".txt", speech_sep, mixed_signal, speech_signal)

end