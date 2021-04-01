using Revise
using SituatedSoundscaping

# prepare data
prepare_data("data/train_speech_raw", "data/train_speech_processed")

# load data
using HDF5
xi = h5read("data/train_speech_processed/"*string(1, pad=10)*".h5", "data_logpower")
xi = h5read("1.h5", "data")

# plot data
using PyPlot
plt.figure()
plt.imshow(xi, aspect="auto", origin="lower", cmap="jet")
plt.gcf()
