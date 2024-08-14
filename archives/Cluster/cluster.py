import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from itertools import combinations

# Re-running the StockCluster class, clustering, and generating pairs

class StockCluster:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.stock_names = data.iloc[:, 0].values
        self.features = data.iloc[:, 1:].values
        scaler = StandardScaler()
        self.scaled_features = scaler.fit_transform(self.features)
        self.labels_ = None
        
    def cluster(self, method='kmeans', n_clusters=2):
        if method == 'kmeans':
            model = KMeans(n_clusters=n_clusters, random_state=42)
        elif method == 'hierarchical':
            model = AgglomerativeClustering(n_clusters=n_clusters)
        else:
            raise ValueError("Method must be 'kmeans' or 'hierarchical'.")
        self.labels_ = model.fit_predict(self.scaled_features)
        
    def get_clusters(self):
        if self.labels_ is None:
            raise ValueError("You need to perform clustering first using the 'cluster' method.")
        cluster_data = pd.DataFrame({
            'Stock': self.stock_names,
            'Cluster': self.labels_
        })
        return cluster_data.groupby('Cluster')['Stock'].apply(list).reset_index()



def generate_pairs_with_year(cluster_df, year):
    """
    Generate all possible pairs within each cluster with individual stock names and the given year.
    
    Parameters:
    - cluster_df: DataFrame with 'Cluster' and 'Stock' columns. Each row should contain a cluster and the stocks within that cluster.
    - year: The year for which clustering is performed.
    
    Returns:
    - DataFrame with 'Pair Name', 'Stock 1', 'Stock 2', 'Cluster', and 'Year' columns.
    """
    pair_data = []
    for idx, row in cluster_df.iterrows():
        cluster = row['Cluster']
        stocks = row['Stock']
        for pair in combinations(stocks, 2):
            pair_name = f"{pair[0]}/{pair[1]}"
            stock_1 = pair[0]
            stock_2 = pair[1]
            pair_data.append([pair_name, stock_1, stock_2, cluster, year])
    
    return pd.DataFrame(pair_data, columns=['Pair Name', 'Stock 1', 'Stock 2', 'Cluster', 'Year'])



