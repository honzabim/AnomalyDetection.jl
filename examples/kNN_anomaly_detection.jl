
using Plots
plotly()
clibrary(:Plots)
using JLD
code_path = "../src/"
push!(LOAD_PATH, code_path)
using AnomalyDetection

dataset = load("toy_data_3.jld")["data"]

X = AnomalyDetection.Float.(dataset.data);
Y = dataset.labels;
nX = X[:, Y.==0]

# model parameters
k = 11 # number of nearest neighbors
metric = Distances.Euclidean() # any of metric from Distance package
distances = "all" # "all"/"last" - use average of all or just the k-th nearest neighbour
contamination = size(Y[Y.==1],1)/size(Y,1)
reduced_dim = false # if dim > 10, use PCA to reduce it
Beta = 1.0
#model = kNN(k, metric = metric, weights = weights, reduced_dim = reduced_dim)
model = kNN(k, contamination, metric = metric, distances = distances,
    reduced_dim = reduced_dim, Beta = Beta)

size(nX)

AnomalyDetection.fit!(model, nX);
AnomalyDetection.setthreshold!(model, X);

# this fits the model and produces predicted labels
tryhat, tstyhat = AnomalyDetection.rocstats(X, Y, X, Y, model);

# plot heatmap of the fit
xl = (minimum(X[1,:])-0.05, maximum(X[1,:]) + 0.05)
yl = (minimum(X[2,:])-0.05, maximum(X[2,:]) + 0.05)
p = scatter(X[1, tryhat.==1], X[2, tryhat.==1], c = :red, label = "predicted positive",
    xlims=xl, ylims = yl, title = "classification results")
scatter!(X[1, tryhat.==0], X[2, tryhat.==0], c = :green, label = "predicted negative",
    legend = (0.7, 0.7))

x = linspace(xl[1], xl[2], 30)
y = linspace(yl[1], yl[2], 30)
zz = zeros(size(y,1),size(x,1))
for i in 1:size(y, 1)
    for j in 1:size(x, 1)
        zz[i,j] = AnomalyDetection.anomalyscore(model, AnomalyDetection.Float.([x[j], y[i]]))
    end
end
contourf!(x, y, zz, c = :viridis)

display(p)
if !isinteractive()
    gui()
end

# plot the roc curve as well
ascore = AnomalyDetection.anomalyscore(model, X);
recvec, fprvec = AnomalyDetection.getroccurve(ascore, Y)

function plotroc(args...)
    # plot the diagonal line
    p = plot(linspace(0,1,100), linspace(0,1,100), c = :gray, alpha = 0.5, xlim = [0,1],
    ylim = [0,1], label = "", xlabel = "false positive rate", ylabel = "true positive rate",
    title = "ROC")
    for arg in args
        plot!(arg[1], arg[2], label = arg[3], lw = 2)
    end
    return p
end

plargs = [(fprvec, recvec, "kNN")]
display(plotroc(plargs...))
if !isinteractive()
    gui()
end
