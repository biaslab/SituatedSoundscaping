using PyCall
using Conda

export SNR, __init__, PESQ, STOI

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