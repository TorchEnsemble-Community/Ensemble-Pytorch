import matplotlib.pyplot as plt

plt.style.use("seaborn")

n_estimators = [2, 5, 7, 10]
fusion = [95.32, 95.34, 95.31, 95.29]
voting = [96.00, 94.77, 96.41, 95.34]
bagging = [95.71, 94.97, 96.16, 94.94]
gradient_boosting = [95.63, 95.78, 95.77, 96.03]
snapshot_ensemble = [95.57, 95.71, 96.34, 95.68]

plt.plot(n_estimators, fusion, label="fusion", linewidth=5)
plt.plot(n_estimators, voting, label="voting", linewidth=5)
plt.plot(n_estimators, bagging, label="bagging", linewidth=5)
plt.plot(n_estimators, gradient_boosting, label="gradient boosting", linewidth=5)
plt.plot(n_estimators, snapshot_ensemble, label="snapshot ensemble", linewidth=5)

plt.title("ResNet@CIFAR-10", fontdict={"size": 30})
plt.xlabel("n_estimators", fontdict={"size": 30})
plt.ylabel("testing acc", fontdict={"size": 30})
plt.xticks(n_estimators, fontsize=25)
plt.yticks(fontsize=25)
plt.legend(prop={"size": 30})
