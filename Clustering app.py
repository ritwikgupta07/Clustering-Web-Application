import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

# Streamlit app
st.title("Clustering Application")

# File upload
uploaded_file = st.file_uploader("Upload an Excel or CSV file", type=["xlsx", "csv"])

if uploaded_file is not None:
    # Read the file based on its extension
    if uploaded_file.name.endswith(".xlsx"):
        data = pd.read_excel(uploaded_file)
    else:  # CSV file
        data = pd.read_csv(uploaded_file)

    # Display the dataframe
    st.write("Data Preview:")
    st.dataframe(data)

    # Select columns for clustering
    col1 = st.selectbox("Select the first column", data.columns)
    col2 = st.selectbox("Select the second column", data.columns)

    # Choose clustering algorithm
    algorithm = st.selectbox("Select Clustering Algorithm", ["K-Means", "Hierarchical"])

    # Select the number of clusters
    n_clusters = st.slider("Select number of clusters", 2, 10, 4)

    if st.button("Perform Clustering"):
        # Create a DataFrame with non-null values
        X = data[[col1, col2]].dropna()

        # Encode non-numeric columns
        if X[col1].dtype == "object" or X[col2].dtype == "object":
            X_encoded = pd.get_dummies(X, columns=[col1, col2], drop_first=True)
        else:
            X_encoded = X.copy()

        # Scale the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_encoded)

        # Clustering
        if algorithm == "K-Means":
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(X_scaled)
        else:  # Hierarchical
            hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
            cluster_labels = hierarchical.fit_predict(X_scaled)

        # Add cluster labels to the original DataFrame
        X["Cluster"] = cluster_labels

        # Plotting
        plt.figure(figsize=(10, 6))
        for cluster in range(n_clusters):
            plt.scatter(
                X[X["Cluster"] == cluster][col2],
                X[X["Cluster"] == cluster][col1],
                label=f"Cluster {cluster + 1}",
                marker="o",
            )

        plt.title(f"{algorithm} Clustering on {col2} and {col1}")
        plt.xlabel(col2)
        plt.ylabel(col1)
        plt.legend(title="Clusters")
        plt.grid()

        # Show the plot in Streamlit
        st.pyplot(plt)
