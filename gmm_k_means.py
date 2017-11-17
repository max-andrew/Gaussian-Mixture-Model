import numpy as np
from matplotlib import pyplot as plt
from sklearn import mixture

dat = np.genfromtxt("gmmData.txt", delimiter=",")

aicsums = np.zeros(6)
bicsums = np.zeros(6)

def plot(scores, color):
	plt.axis([1, 6, 1500, 3000])
	plt.plot([1,2,3,4,5,6], scores, color, label="AIC" if color == 'r' else "BIC")

for x in range(1,7):
	for y in range(100):
		gmm = mixture.GaussianMixture(n_components=x)
		gmm.fit(dat)
		aicsums[x-1] += gmm.aic(dat)
		bicsums[x-1] += gmm.bic(dat)

aicsums = np.divide(aicsums, 100)
bicsums = np.divide(bicsums, 100)

plot(aicsums, 'r')
plot(bicsums, 'b')
plt.legend()
plt.show()