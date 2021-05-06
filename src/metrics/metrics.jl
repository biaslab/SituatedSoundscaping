using PyCall
using Conda

export SNR, __init__, PESQ, STOI, evaluate_metrics

# install conda packages
Conda.pip_interop(true)
Conda.pip("install", "pesq")
Conda.pip("install", "pystoi")

# signal to noise ratio
SNR(y_true, y_reconstructed) = 10*log10(mean(abs2.(y_true)) ./ mean(abs2.(y_true - y_reconstructed)))

# https://pypi.org/project/pesq/
const PESQ = PyNULL()

# https://github.com/mpariente/pystoi
const STOI = PyNULL()

function __init__() 
    copy!(PESQ, pyimport("pesq").pesq)
    copy!(STOI, pyimport("pystoi").stoi)
end

function evaluate_metrics(filename, speech_sep, speech_noisy, speech_true; block_length=32::Int, fs=16000::Real)

    # also 'warp' the original signal
    filterbank_true = warped_filter_bank(block_duration_s=block_length/fs, nr_bands=Int(block_length/2)+1)
    filterbank_noisy = warped_filter_bank(block_duration_s=block_length/fs, nr_bands=Int(block_length/2)+1)
    nr_blocks = Int(length(speech_true)/block_length)
    speech_true_warped = zeros(size(speech_true))
    speech_noisy_warped = zeros(size(speech_noisy))
    for k in 1:nr_blocks

        # feed signals into filterbanks
        run!(filterbank_true, speech_true[1+(k-1)*block_length:k*block_length])
        run!(filterbank_noisy, speech_noisy[1+(k-1)*block_length:k*block_length])

        # read from filter
        speech_true_warped[1+(k-1)*block_length:k*block_length] = read(filterbank_true)
        speech_noisy_warped[1+(k-1)*block_length:k*block_length] = read(filterbank_noisy)

    end

    # calculate metrics
    SNRo = SNR(speech_true_warped, speech_sep)
    pesqw = PESQ(fs, speech_true_warped, speech_sep, "wb")
    pesqn = PESQ(fs, speech_true_warped, speech_sep, "nb")
    stoi = STOI(speech_true_warped, speech_sep, fs, extended=false)

    # calculate reference metrics
    SNR_baseline = SNR(speech_true_warped, speech_noisy_warped)
    pesqw_baseline = PESQ(fs, speech_true_warped, speech_noisy_warped, "wb")
    pesqn_baseline = PESQ(fs, speech_true_warped, speech_noisy_warped, "nb")
    stoi_baseline = STOI(speech_true_warped, speech_noisy_warped, fs, extended=false)

    # save metrics
    open(filename, "w") do f
        write(f, "BASELINE \n")
        write(f, "-------- \n")
        write(f, "SNR = $SNR_baseline \n")
        write(f, "PESQ (wb)= $pesqw_baseline \n")
        write(f, "PESQ (nb)= $pesqn_baseline \n")
        write(f, "STOI= $stoi_baseline \n")
        write(f, "\n")
        write(f, "\n")
        write(f, "RESULTS \n")
        write(f, "-------- \n")
        write(f, "SNR = $SNRo \n")
        write(f, "PESQ (wb)= $pesqw \n")
        write(f, "PESQ (nb)= $pesqn \n")
        write(f, "STOI= $stoi \n")
    end

end
