''''
Standalone file for the k-means clustering
'''

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

def is_int(val):
    try:
        num = int(val)
    except ValueError:
        return False
    return True


def mode(a):
    u, c = np.unique(a, return_counts=True)
    return u[c.argmax()]


mnt = r'Z:/'
FIN = mnt+'data/2d_PTS_pars_static_filtered.csv'
FOUT = mnt+'data/2d_PTS_pars_static_filtered_k-means_clusters.csv'

df = pd.read_csv(FIN)
df = df.set_index('RGI code ')

dfnames = pd.DataFrame(df.columns)

# ---- k-means clustering -----#
# nc - number of clusters
# -------------- USING STD
dfc = pd.DataFrame([df['std per gl']]).T
nc = 3
kmeans = KMeans(n_clusters=nc).fit(dfc)
z = kmeans.labels_
df['cluster3_mbstd'] = z
nc = 4
kmeans = KMeans(n_clusters=nc).fit(dfc)
z = kmeans.labels_
df['cluster4_mbstd'] = z
nc = 5
kmeans = KMeans(n_clusters=nc).fit(dfc)
z = kmeans.labels_
df['cluster5_mbstd'] = z

# -------------- USING SMB
dfc = pd.DataFrame([df['mb']]).T
nc = 3
kmeans = KMeans(n_clusters=nc).fit(dfc)
z = kmeans.labels_
df['cluster3_mb'] = z
nc = 4
kmeans = KMeans(n_clusters=nc).fit(dfc)
z = kmeans.labels_
df['cluster4_mb'] = z
nc = 5
kmeans = KMeans(n_clusters=nc).fit(dfc)
z = kmeans.labels_
df['cluster5_mb'] = z

# write output - all in one file
df.to_csv(FOUT)
