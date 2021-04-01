using Revise
using SituatedSoundscaping

# settings 
nr_mixtures = 100
nr_files = 10
nr_iterations_em = 10
nr_iterations_gs = 10

# prepare data
data = prepare_data("data/train_speech_raw", "data/train_speech_processed")

# train Kmeans model
centers, πk1 = train_kmeans("models/Kmeans/speech", data[1:nr_files], nr_mixtures)

# train EM model
means, covs, πk2 = train_em("models/Kmeans/speech", data[1:nr_files], centers, πk1; nr_iterations=nr_iterations_em)

# train GS model



# load data
using HDF5
xi = h5read("data/train_speech_processed/"*string(1, pad=10)*".h5", "data_logpower")
xi = h5read("1.h5", "data")

# plot data
using PyPlot
plt.figure()
plt.imshow(xi[:,1:1000], aspect="auto", origin="lower", cmap="jet")
plt.gcf()
