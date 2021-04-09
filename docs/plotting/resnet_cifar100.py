import matplotlib.pyplot as plt

plt.style.use("seaborn")

n_estimators = [2, 5, 7, 10]
fusion = [77.14, 76.81, 76.52, 76.06]
voting = [78.04, 79.39, 79.63, 79.78]
bagging = [77.21, 78.94, 79.09, 79.47]
snapshot_ensemble = [77.30, 77.39, 77.91, 77.58]

plt.plot(n_estimators, fusion, label="fusion", linewidth=5)
plt.plot(n_estimators, voting, label="voting", linewidth=5)
plt.plot(n_estimators, bagging, label="bagging", linewidth=5)
plt.plot(n_estimators, snapshot_ensemble, label="snapshot ensemble", linewidth=5)

plt.title("ResNet@CIFAR-100", fontdict={"size": 30})
plt.xlabel("n_estimators", fontdict={"size": 30})
plt.ylabel("testing acc", fontdict={"size": 30})
plt.xticks(n_estimators, fontsize=25)
plt.yticks(fontsize=25)
plt.legend(prop={"size": 30})
