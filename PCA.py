import random as rd

import numpy as np
import pandas as pd

gen =['gen'+ str(i) for i in range(1,101)]
wt =['wt'+ str(i) for i in range(1,6)]
ko =['ko'+ str(i) for i in range(1,6)]
data = pd.DataFrame(columns=[*wt,*ko],index=gen)

for gen in data.index:
    data.loc[gen,'wt1':'wt5'] = np.random.poisson(lam=rd.randrange(1,1000),size=5)
    data.loc[gen, 'ko1':'ko5'] = np.random.poisson(lam=rd.randrange(1, 1000), size=5)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
scaled_data = StandardScaler().fit_transform(data.T)
pca = PCA()
pca.fit(scaled_data)
pca_data = pca.transform(scaled_data)
per_var = np.round(pca.explained_variance_ratio_* 100,decimals=1)
labels = ['PC' + str(x) for x in range(1, len(per_var)+1)]
import matplotlib.pyplot as plt
plt.bar(x=range(1,len(per_var)+1),height=per_var,tick_label=labels)
plt.ylabel('Percentage of Eplained Variance')
plt.xlabel('Principl Component')
plt.title('Scree Plot')
