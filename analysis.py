import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import plotly.graph_objs as go
from plotly.offline import plot

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

# Function to perform clustering within each category
def cluster_category(category_data, category_name, n_clusters):
    features = ['Total_Venta', 'Total_Utilidad', 'Volumen', 'PesoBruto']
    category_features = category_data[features]
    
    # Standardize the features
    scaler = StandardScaler()
    category_features_scaled = scaler.fit_transform(category_features)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    category_data = category_data.copy()
    category_data['Cluster'] = kmeans.fit_predict(category_features_scaled)
    
    # Perform PCA
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(category_features_scaled)
    category_data['PC1'] = pca_result[:, 0]
    category_data['PC2'] = pca_result[:, 1]
    category_data['PC3'] = pca_result[:, 2]
    
    return category_data

# Cluster each category
small_clients = cluster_category(client_data[client_data['Category'] == 'Small'], 'Small', 4)
medium_clients = cluster_category(client_data[client_data['Category'] == 'Medium'], 'Medium', 3)
big_clients = cluster_category(client_data[client_data['Category'] == 'Big'], 'Big', 3)

# Function to create 3D scatter plot for each client category
def create_3d_scatter_plot(clustered_data, category_name):
    traces = []
    colors = ['rgba(255, 128, 255, 0.8)', 'rgba(255, 128, 2, 0.8)', 'rgba(0, 255, 200, 0.8)', 'rgba(0, 128, 255, 0.8)']
    legend_text = []

    for cluster in clustered_data['Cluster'].unique():
        cluster_data = clustered_data[clustered_data['Cluster'] == cluster]
        trace = go.Scatter3d(
            x=cluster_data['PC1'],
            y=cluster_data['PC2'],
            z=cluster_data['PC3'],
            mode='markers',
            marker=dict(size=5, color=colors[cluster % len(colors)]),
            name=f'Cluster {cluster}',
            text=cluster_data['Cliente'],
            hoverinfo='text',
            hovertext=(
                'Cliente: ' + cluster_data['Cliente'] + '<br>' +
                'Total Venta: ' + cluster_data['Total_Venta'].astype(str) + '<br>' +
                'Total Utilidad: ' + cluster_data['Total_Utilidad'].astype(str) + '<br>' +
                'Volumen: ' + cluster_data['Volumen'].astype(str)
            )
        )
        traces.append(trace)

    layout = go.Layout(
        title=f'Plot of {category_name} Clients Clusters',
        scene=dict(
            xaxis=dict(title='Total Venta'),
            yaxis=dict(title='Total Utilidad'),
            zaxis=dict(title='Volumen'),
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        showlegend=False 
    )

    fig = go.Figure(data=traces, layout=layout)
    plot(fig, filename=f'{category_name.lower()}_clients_clusters.html')

def print_cluster_clients(clustered_data, category_name):
    for cluster in clustered_data['Cluster'].unique():
        cluster_data = clustered_data[clustered_data['Cluster'] == cluster]
        clients_list = cluster_data['Cliente'].tolist()
        print(f'\nClients in {category_name} Cluster {cluster}:')
        for client in clients_list:
            print(client)
            
# Create plots for each category
create_3d_scatter_plot(small_clients, 'Small')
print_cluster_clients(small_clients, 'Small')

create_3d_scatter_plot(medium_clients, 'Medium')
print_cluster_clients(medium_clients, 'Medium')

create_3d_scatter_plot(big_clients, 'Big')
print_cluster_clients(big_clients, 'Big')
small_clients.to_csv(r'C:\Users\Roberto H\logistics\small_clients_clusters.csv', index=False)
medium_clients.to_csv(r'C:\Users\Roberto H\logistics\medium_clients_clusters.csv', index=False)
big_clients.to_csv(r'C:\Users\Roberto H\logistics\big_clients_clusters.csv', index=False)
