import matplotlib.pyplot as plt

plt.style.use("seaborn")

n_estimators = [5, 10, 15, 20]
fusion = [99.16, 99.24, 99.20, 99.23]
voting = [99.47, 99.47, 99.51, 99.52]
bagging = [99.47, 99.52, 99.52, 99.54]
gradient_boosting = [99.13, 99.53, 99.51, 99.43]
snapshot_ensemble = [99.22, 99.25, 99.28, 99.32]

plt.plot(n_estimators, fusion, label="fusion", linewidth=5)
plt.plot(n_estimators, voting, label="voting", linewidth=5)
plt.plot(n_estimators, bagging, label="bagging", linewidth=5)
plt.plot(n_estimators, gradient_boosting, label="gradient boosting", linewidth=5)
plt.plot(n_estimators, snapshot_ensemble, label="snapshot ensemble", linewidth=5)

plt.title("LeNet@MNIST", fontdict={"size": 30})
plt.xlabel("n_estimators", fontdict={"size": 30})
plt.ylabel("testing acc", fontdict={"size": 30})
plt.xticks(n_estimators, fontsize=25)
plt.yticks(fontsize=25)
plt.legend(prop={"size": 30})
