import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Step 2: Load dataset
df = pd.read_csv("customers.csv")

# Step 3: Preprocess Data
# Encode categorical features
if 'Gender' in df.columns:
    le = LabelEncoder()
    df['Gender'] = le.fit_transform(df['Gender'])  # Male=1, Female=0

# Select relevant features
features = ['Age', 'Gender', 'Annual Income (k$)', 'Spending Score (1-100)']
X = df[features]

# Step 4: Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 5: Determine optimal number of clusters using Elbow Method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method - Optimal K')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.grid(True)
plt.show()

# Step 6: Apply KMeans with optimal number of clusters (e.g., 5)
k_optimal = 5
kmeans = KMeans(n_clusters=k_optimal, init='k-means++', random_state=42)
clusters = kmeans.fit_predict(X_scaled)
df['Cluster'] = clusters

# Step 7: Visualize clusters with PCA (2D projection)
pca = PCA(n_components=2)
pca_components = pca.fit_transform(X_scaled)
df['PCA1'] = pca_components[:, 0]
df['PCA2'] = pca_components[:, 1]

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Cluster', palette='Set1', s=100, alpha=0.7)
plt.title("Customer Segments Visualization")
plt.grid(True)
plt.show()

# Step 8: Analyze cluster characteristics
cluster_summary = df.groupby('Cluster')[features].mean()
print("Cluster-wise Customer Profile Summary:")
print(cluster_summary)

# Step 9: Save clustered customer data
df.to_csv("clustered_customers.csv", index=False)
print("Clustered data saved as 'clustered_customers.csv'.")
