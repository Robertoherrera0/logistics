import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the data from the Excel file
file_path = r'C:\Users\Roberto H\logistics\DETALLE DE ORDEN POR CLIENTE 2023.xlsx'
data = pd.read_excel(file_path)

# Fill missing values with the mean of their respective columns for numerical data
data.fillna(data.mean(numeric_only=True), inplace=True)

# For categorical columns, fill missing values with the most frequent value
data.fillna(data.mode().iloc[0], inplace=True)

# Aggregate data by clients
client_data = data.groupby('Cliente').agg({
    'Total_Venta': 'sum',
    'Total_Utilidad': 'sum',
    'Volumen': 'sum',
    'PesoBruto': 'sum'
}).reset_index()

# Exclude any 'Total' row from the data
client_data = client_data[client_data['Cliente'] != 'Total']

# Function to perform clustering within each category
def cluster_category(category_data, category_name):
    features = ['Total_Venta', 'Total_Utilidad', 'Volumen', 'PesoBruto']
    category_features = category_data[features]
    
    # Standardize the features
    scaler = StandardScaler()
    category_features_scaled = scaler.fit_transform(category_features)
    
    if category_name == 'Big':
        optimal_clusters = 3
    elif category_name == "Medium":
        optimal_clusters = 3
    else:
        optimal_clusters = 5

    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
    category_data = category_data.copy()
    category_data['Cluster'] = kmeans.fit_predict(category_features_scaled)
    
    return category_data

# Segment clients into small, medium, and big based on Total_Venta
quantiles = client_data['Total_Venta'].quantile([0.33, 0.66])

def categorize_client(total_venta):
    if total_venta <= quantiles[0.33]:
        return 'Small'
    elif total_venta <= quantiles[0.66]:
        return 'Medium'
    else:
        return 'Big'

client_data['Category'] = client_data['Total_Venta'].apply(categorize_client)

# Cluster each category
small_clients = client_data[client_data['Category'] == 'Small']
medium_clients = client_data[client_data['Category'] == 'Medium']
big_clients = client_data[client_data['Category'] == 'Big']

small_clients = cluster_category(small_clients, 'Small')
medium_clients = cluster_category(medium_clients, 'Medium')
big_clients = cluster_category(big_clients, 'Big')

# Combine the clustered data
clustered_data = pd.concat([small_clients, medium_clients, big_clients])

# Analyze the clusters
def analyze_clusters(clustered_data, category_name):
    cluster_analysis = clustered_data.groupby('Cluster').agg({
        'Total_Venta': 'mean',
        'Total_Utilidad': 'mean',
        'Volumen': 'mean',
        'PesoBruto': 'mean',
        'Cliente': 'count'
    }).rename(columns={'Cliente': 'Number_of_Clients'})
    print(f"\nCluster Analysis for {category_name} Clients:\n", cluster_analysis)
    
    # Print client names for each cluster
    for cluster_label in clustered_data['Cluster'].unique():
        clients_in_cluster = clustered_data[clustered_data['Cluster'] == cluster_label]['Cliente'].tolist()
        print(f"\nClients in Cluster {cluster_label} ({category_name} Clients):\n", clients_in_cluster)

# Analyze clusters for each category
analyze_clusters(small_clients, 'Small')
analyze_clusters(medium_clients, 'Medium')
analyze_clusters(big_clients, 'Big')

# Save the clustered data to a CSV file
output_file_path = r'C:\Users\Roberto H\logistics\client_clusters_including_highest.csv'
clustered_data.to_csv(output_file_path, index=False)

# Visualizing Clusters for Small Clients
plt.figure(figsize=(12, 8))
sns.scatterplot(data=small_clients, x='Total_Venta', y='Total_Utilidad', hue='Cluster', palette='viridis', style='Cluster', markers=True)
plt.title('Clusters of Small Clients based on Total_Venta and Total_Utilidad')
plt.xlabel('Total_Venta')
plt.ylabel('Total_Utilidad')
plt.legend(title='Cluster')
plt.show()

# Visualizing Clusters for Medium Clients
plt.figure(figsize=(12, 8))
sns.scatterplot(data=medium_clients, x='Total_Venta', y='Total_Utilidad', hue='Cluster', palette='viridis', style='Cluster', markers=True)
plt.title('Clusters of Medium Clients based on Total_Venta and Total_Utilidad')
plt.xlabel('Total_Venta')
plt.ylabel('Total_Utilidad')
plt.legend(title='Cluster')
plt.show()

# Visualizing Clusters for Big Clients
plt.figure(figsize=(12, 8))
sns.scatterplot(data=big_clients, x='Total_Venta', y='Total_Utilidad', hue='Cluster', palette='viridis', style='Cluster', markers=True)
plt.title('Clusters of Big Clients based on Total_Venta and Total_Utilidad')
plt.xlabel('Total_Venta')
plt.ylabel('Total_Utilidad')
plt.legend(title='Cluster')
plt.show()

# Analyzing Volumen and PesoBruto for Small Clients
plt.figure(figsize=(12, 8))
sns.boxplot(data=small_clients, x='Cluster', y='Volumen', palette='viridis')
plt.title('Distribution of Volumen in Small Clients Clusters')
plt.xlabel('Cluster')
plt.ylabel('Volumen')
plt.show()

plt.figure(figsize=(12, 8))
sns.boxplot(data=small_clients, x='Cluster', y='PesoBruto', palette='viridis')
plt.title('Distribution of PesoBruto in Small Clients Clusters')
plt.xlabel('Cluster')
plt.ylabel('PesoBruto')
plt.show()

# Analyzing Volumen and PesoBruto for Medium Clients
plt.figure(figsize=(12, 8))
sns.boxplot(data=medium_clients, x='Cluster', y='Volumen', palette='viridis')
plt.title('Distribution of Volumen in Medium Clients Clusters')
plt.xlabel('Cluster')
plt.ylabel('Volumen')
plt.show()

plt.figure(figsize=(12, 8))
sns.boxplot(data=medium_clients, x='Cluster', y='PesoBruto', palette='viridis')
plt.title('Distribution of PesoBruto in Medium Clients Clusters')
plt.xlabel('Cluster')
plt.ylabel('PesoBruto')
plt.show()

# Analyzing Volumen and PesoBruto for Big Clients
plt.figure(figsize=(12, 8))
sns.boxplot(data=big_clients, x='Cluster', y='Volumen', palette='viridis')
plt.title('Distribution of Volumen in Big Clients Clusters')
plt.xlabel('Cluster')
plt.ylabel('Volumen')
plt.show()

plt.figure(figsize=(12, 8))
sns.boxplot(data=big_clients, x='Cluster', y='PesoBruto', palette='viridis')
plt.title('Distribution of PesoBruto in Big Clients Clusters')
plt.xlabel('Cluster')
plt.ylabel('PesoBruto')
plt.show()
