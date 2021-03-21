import matplotlib.pyplot as plt

plt.style.use("seaborn")

n_estimators = [5, 10, 15, 20]
fusion = [78.26, 78.79, 78.44, 77.20]
voting = [77.71, 78.22, 78.13, 78.73]
bagging = [77.15, 77.32, 77.52, 77.85]
gradient_boosting = [77.28, 78.88, 79.19, 78.60]
snapshot_ensemble = [70.83, 70.69, 70.06, 70.37]

plt.plot(n_estimators, fusion, label="fusion", linewidth=5)
plt.plot(n_estimators, voting, label="voting", linewidth=5)
plt.plot(n_estimators, bagging, label="bagging", linewidth=5)
plt.plot(n_estimators, gradient_boosting, label="gradient boosting", linewidth=5)
plt.plot(n_estimators, snapshot_ensemble, label="snapshot ensemble", linewidth=5)

plt.title("LeNet@CIFAR-10", fontdict={"size": 30})
plt.xlabel("n_estimators", fontdict={"size": 30})
plt.ylabel("testing acc", fontdict={"size": 30})
plt.xticks(n_estimators, fontsize=25)
plt.yticks(fontsize=25)
plt.legend(prop={"size": 30})
