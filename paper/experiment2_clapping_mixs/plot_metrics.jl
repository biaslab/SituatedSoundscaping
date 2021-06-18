using HDF5
using PGFPlotsX


# fetch results
begin
    metrics_algonquin = Dict("SNR"=> Dict("x"=> zeros(5), "y"=> zeros(5)), "PESQ_nb"=> Dict("x"=> zeros(5), "y"=> zeros(5)), "PESQ_wb"=> Dict("x"=> zeros(5), "y"=> zeros(5)), "STOI"=> Dict("x"=> zeros(5), "y"=> zeros(5)))
    metrics_baseline = Dict("SNR"=> Dict("x"=> zeros(5), "y"=> zeros(5)), "PESQ_nb"=> Dict("x"=> zeros(5), "y"=> zeros(5)), "PESQ_wb"=> Dict("x"=> zeros(5), "y"=> zeros(5)), "STOI"=> Dict("x"=> zeros(5), "y"=> zeros(5)))
    mixs_all = sort(map(x -> parse(Int64, split(x[findfirst("mixs=", x)[end]+1:end], "_")[1]), filter(x -> occursin(".h5", x) & occursin("metrics",x), readdir("paper/experiment2_clapping_mixs/exports/algonquin_vb",join=true))))
    for metrics_file in filter(x -> occursin(".h5", x) & occursin("metrics",x), readdir("paper/experiment2_clapping_mixs/exports/algonquin_vb",join=true))
        mixs = parse(Int64, split(metrics_file[findfirst("mixs=", metrics_file)[end]+1:end], "_")[1])
        mixs_ind = findall(x -> x==mixs, mixs_all)[1]
        metrics_algonquin["SNR"]["y"][mixs_ind] = h5read(metrics_file, "new_SNR")
        metrics_algonquin["SNR"]["x"][mixs_ind] = mixs
        metrics_algonquin["PESQ_nb"]["y"][mixs_ind] = h5read(metrics_file, "new_PESQnb")
        metrics_algonquin["PESQ_nb"]["x"][mixs_ind] = mixs
        metrics_algonquin["PESQ_wb"]["y"][mixs_ind] = h5read(metrics_file, "new_PESQwb")
        metrics_algonquin["PESQ_wb"]["x"][mixs_ind] = mixs
        metrics_algonquin["STOI"]["y"][mixs_ind] = h5read(metrics_file, "new_STOI")
        metrics_algonquin["STOI"]["x"][mixs_ind] = mixs
        metrics_baseline["SNR"]["y"][mixs_ind] = h5read(metrics_file, "baseline_SNR")
        metrics_baseline["SNR"]["x"][mixs_ind] = mixs
        metrics_baseline["PESQ_nb"]["y"][mixs_ind] = h5read(metrics_file, "baseline_PESQnb")
        metrics_baseline["PESQ_nb"]["x"][mixs_ind] = mixs
        metrics_baseline["PESQ_wb"]["y"][mixs_ind] = h5read(metrics_file, "baseline_PESQwb")
        metrics_baseline["PESQ_wb"]["x"][mixs_ind] = mixs
        metrics_baseline["STOI"]["y"][mixs_ind] = h5read(metrics_file, "baseline_STOI")
        metrics_baseline["STOI"]["x"][mixs_ind] = mixs
    end

    # fetch metrics gs_sum
    metrics_gs_sum = Dict("SNR"=> Dict("x"=> zeros(5), "y"=> zeros(5)), "PESQ_nb"=> Dict("x"=> zeros(5), "y"=> zeros(5)), "PESQ_wb"=> Dict("x"=> zeros(5), "y"=> zeros(5)), "STOI"=> Dict("x"=> zeros(5), "y"=> zeros(5)))
    mixs_all = sort(map(x -> parse(Int64, split(x[findfirst("mixs=", x)[end]+1:end], "_")[1]), filter(x -> occursin(".h5", x) & occursin("metrics",x), readdir("paper/experiment2_clapping_mixs/exports/gs_sum",join=true))))
    for metrics_file in filter(x -> occursin(".h5", x) & occursin("metrics",x), readdir("paper/experiment2_clapping_mixs/exports/gs_sum",join=true))
        mixs = parse(Int64, split(metrics_file[findfirst("mixs=", metrics_file)[end]+1:end], "_")[1])
        mixs_ind = findall(x -> x==mixs, mixs_all)[1]
        metrics_gs_sum["SNR"]["y"][mixs_ind] = h5read(metrics_file, "new_SNR")
        metrics_gs_sum["SNR"]["x"][mixs_ind] = mixs
        metrics_gs_sum["PESQ_nb"]["y"][mixs_ind] = h5read(metrics_file, "new_PESQnb")
        metrics_gs_sum["PESQ_nb"]["x"][mixs_ind] = mixs
        metrics_gs_sum["PESQ_wb"]["y"][mixs_ind] = h5read(metrics_file, "new_PESQwb")
        metrics_gs_sum["PESQ_wb"]["x"][mixs_ind] = mixs
        metrics_gs_sum["STOI"]["y"][mixs_ind] = h5read(metrics_file, "new_STOI")
        metrics_gs_sum["STOI"]["x"][mixs_ind] = mixs
    end

    # fetch metrics wiener
    metrics_wiener = Dict("SNR"=> Dict("x"=> zeros(5), "y"=> zeros(5)), "PESQ_nb"=> Dict("x"=> zeros(5), "y"=> zeros(5)), "PESQ_wb"=> Dict("x"=> zeros(5), "y"=> zeros(5)), "STOI"=> Dict("x"=> zeros(5), "y"=> zeros(5)))
    metrics_file = "paper/experiment2_clapping_mixs/exports/wiener/metrics_power=0.h5"
    for mixs_ind = 1:length(mixs_all)
        metrics_wiener["SNR"]["y"][mixs_ind] = h5read(metrics_file, "new_SNR")
        metrics_wiener["SNR"]["x"][mixs_ind] = mixs_all[mixs_ind]
        metrics_wiener["PESQ_nb"]["y"][mixs_ind] = h5read(metrics_file, "new_PESQnb")
        metrics_wiener["PESQ_nb"]["x"][mixs_ind] = mixs_all[mixs_ind]
        metrics_wiener["PESQ_wb"]["y"][mixs_ind] = h5read(metrics_file, "new_PESQwb")
        metrics_wiener["PESQ_wb"]["x"][mixs_ind] = mixs_all[mixs_ind]
        metrics_wiener["STOI"]["y"][mixs_ind] = h5read(metrics_file, "new_STOI")
        metrics_wiener["STOI"]["x"][mixs_ind] = mixs_all[mixs_ind]
    end
end


plt_metrics = @pgf GroupPlot(
    # group plot options
    {
        group_style = {
            group_size="3 by 1",
            horizontal_sep = "1.5cm",
        },
    },

    # axis 1 (SNR)
    {
        xlabel="number of mixtures (speech)",
        ylabel="output SNR",
        grid = "major",
        style = {thick},
    },
    # plots for axis 1
    Plot(Table(metrics_baseline["SNR"]["x"], metrics_baseline["SNR"]["y"])), LegendEntry("Baseline"),
    Plot(Table(metrics_algonquin["SNR"]["x"], metrics_algonquin["SNR"]["y"])), LegendEntry("Algonquin"),
    Plot(Table(metrics_gs_sum["SNR"]["x"], metrics_gs_sum["SNR"]["y"])), LegendEntry("GS sum"),
    Plot(Table(metrics_wiener["SNR"]["x"], metrics_wiener["SNR"]["y"])), LegendEntry("Wiener"),
    
    # axis 2 (PESQ)
    {
        xlabel="number of mixtures (speech)",
        ylabel="output PESQ",
        grid = "major",
        style = {thick},
    },
    # plots for axis 2
    Plot(Table(metrics_baseline["PESQ_wb"]["x"], metrics_baseline["PESQ_wb"]["y"])), LegendEntry("Baseline"),
    Plot(Table(metrics_algonquin["PESQ_wb"]["x"], metrics_algonquin["PESQ_wb"]["y"])), LegendEntry("Algonquin"),
    Plot(Table(metrics_gs_sum["PESQ_wb"]["x"], metrics_gs_sum["PESQ_wb"]["y"])), LegendEntry("GS sum"),
    Plot(Table(metrics_wiener["PESQ_wb"]["x"], metrics_wiener["PESQ_wb"]["y"])), LegendEntry("Wiener"),

    # axis 3 (STOI)
    { 
        xlabel="number of mixtures (speech)",
        ylabel="output STOI",
        grid = "major",
        style = {thick},
    },
    # plots for axis 3
    Plot(Table(metrics_baseline["STOI"]["x"], metrics_baseline["STOI"]["y"])), LegendEntry("Baseline"),
    Plot(Table(metrics_algonquin["STOI"]["x"], metrics_algonquin["STOI"]["y"])), LegendEntry("Algonquin"),
    Plot(Table(metrics_gs_sum["STOI"]["x"], metrics_gs_sum["STOI"]["y"])), LegendEntry("GS sum"),
    Plot(Table(metrics_wiener["STOI"]["x"], metrics_wiener["STOI"]["y"])), LegendEntry("Wiener"),

)

pgfsave("paper/experiment2_clapping_mixs/exports/figures/metrics.tikz", plt_metrics)