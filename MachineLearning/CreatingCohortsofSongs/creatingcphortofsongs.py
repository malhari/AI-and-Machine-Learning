# =============================
# 1. Import Libraries
# =============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

sns.set(style="whitegrid")

# =============================
# 2. Load Dataset
# =============================
df = pd.read_csv("rolling_stones_spotify.csv")

print(df.head())
print(df.info())

# =============================
# 3. Data Cleaning
# =============================

# Missing values
print("\nMissing Values:\n", df.isnull().sum())

# Remove duplicates
df = df.drop_duplicates()

# Convert release_date
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')

# Duration in minutes (feature engineering)
df['duration_min'] = df['duration_ms'] / 60000

# =============================
# 4. Exploratory Data Analysis
# =============================

# Popularity distribution
plt.figure(figsize=(6,4))
sns.histplot(df['popularity'], bins=20, kde=True)
plt.title("Popularity Distribution")
plt.show()

# Correlation heatmap
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# =============================
# 5. Album Recommendation
# =============================

album_pop = (
    df.groupby('album')['popularity']
    .agg(['count','mean'])
    .sort_values('mean', ascending=False)
)

print("\nTop Albums by Avg Popularity:\n", album_pop.head())

# Visualization
plt.figure(figsize=(10,5))
album_pop['mean'].head(10).plot(kind='bar')
plt.title("Top Albums by Average Popularity")
plt.show()

# =============================
# 6. Popularity vs Features
# =============================

features = [
    'danceability','energy','acousticness',
    'valence','tempo','loudness'
]

for f in features:
    plt.figure(figsize=(5,4))
    sns.scatterplot(x=df[f], y=df['popularity'])
    plt.title(f"Popularity vs {f}")
    plt.show()

# =============================
# 7. Dimensionality Reduction (PCA)
# =============================

audio_features = [
    'acousticness','danceability','energy',
    'instrumentalness','liveness','loudness',
    'speechiness','tempo','valence','duration_min'
]

X = df[audio_features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA()
pca.fit(X_scaled)

plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("Number of Components")
plt.ylabel("Explained Variance")
plt.title("PCA Explained Variance")
plt.show()

# Keep 3 components
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

# =============================
# 8. Find Optimal Clusters
# =============================

wcss = []
sil_scores = []

for k in range(2,11):
    km = KMeans(n_clusters=k, random_state=42)
    labels = km.fit_predict(X_pca)
    wcss.append(km.inertia_)
    sil_scores.append(silhouette_score(X_pca, labels))

# Elbow plot
plt.plot(range(2,11), wcss)
plt.title("Elbow Method")
plt.xlabel("K")
plt.ylabel("WCSS")
plt.show()

# Silhouette plot
plt.plot(range(2,11), sil_scores)
plt.title("Silhouette Scores")
plt.xlabel("K")
plt.ylabel("Score")
plt.show()

# =============================
# 9. Final Clustering
# =============================

kmeans = KMeans(n_clusters=4, random_state=42)
df['cluster'] = kmeans.fit_predict(X_pca)

# =============================
# 10. Cluster Analysis
# =============================

cluster_summary = df.groupby('cluster')[audio_features + ['popularity']].mean()
print("\nCluster Summary:\n", cluster_summary)

# Visualization
plt.figure(figsize=(6,5))
sns.scatterplot(
    x=X_pca[:,0],
    y=X_pca[:,1],
    hue=df['cluster'],
    palette="Set2"
)
plt.title("Song Cohorts (Clusters)")
plt.show()

# =============================
# 11. Interpretation Guide
# =============================

print("""
Cluster Interpretation Tips:

Cluster with:
- High danceability + valence → Happy/Dance tracks
- High energy + loudness → Rock/High-intensity songs
- High acousticness → Acoustic/Soft songs
- High instrumentalness → Instrumental tracks

Use these clusters for recommendation cohorts.
""")
