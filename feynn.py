import pandas as pd

# Load the dataset
mcdonalds = pd.read_csv("https://vincentarelbundock.github.io/Rdatasets/csv/MSA/mcdonalds.csv")

# Display column names
print(mcdonalds.columns)

# Display dimensions of the dataset
print(mcdonalds.shape)

# Display first 3 rows
print(mcdonalds.head(3))

# Select columns 1 to 11 and convert "Yes" to 1 and other values to 0
MD_x = mcdonalds.iloc[:, 0:11].apply(lambda x: (x == "Yes").astype(int))

# Compute column means
col_means = round(MD_x.mean(), 2)
print(col_means)
from sklearn.decomposition import PCA
import numpy as np

# Perform PCA
MD_pca = PCA().fit(MD_x)

# Summary of PCA results
print("Importance of components:")
print(pd.DataFrame({
    "Standard deviation": np.sqrt(MD_pca.explained_variance_).round(4),
    "Proportion of Variance": np.round(MD_pca.explained_variance_ratio_, 4),
    "Cumulative Proportion": np.round(np.cumsum(MD_pca.explained_variance_ratio_), 4)
}, index=[f"PC{i+1}" for i in range(len(MD_pca.explained_variance_))]))

# Printing standard deviations of principal components
print("Standard deviations (1, .., p=11):")
print(np.round(MD_pca.singular_values_, 1))

# Printing rotation matrix
print("Rotation (n x k) = (11 x 11):")
print(pd.DataFrame(MD_pca.components_.T, index=MD_x.columns, columns=[f"PC{i+1}" for i in range(MD_pca.components_.shape[0])]).round(3))

import matplotlib.pyplot as plt

# Project data onto principal components
MD_pca_proj = MD_pca.transform(MD_x)

# Plot data points
plt.scatter(MD_pca_proj[:, 0], MD_pca_proj[:, 1], color='grey')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA Projection')
plt.show()
from sklearn.cluster import KMeans

# Set random seed
import numpy as np
np.random.seed(1234)

# Perform k-means clustering for k = 2 to 8
results = {}
for k in range(2, 9):
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=1234)
    kmeans.fit(MD_x)
    results[k] = kmeans.labels_

# Relabel clusters
# Note: In scikit-learn, labels are already assigned after fitting
# So, relabeling is not necessary
import matplotlib.pyplot as plt

# Plotting the number of segments (k) against within-cluster sum of squares (WCSS)
plt.plot(range(2, 9), [kmeans.inertia_ for kmeans in results.values()], marker='o')
plt.xlabel('Number of segments')
plt.ylabel('Within-cluster sum of squares (WCSS)')
plt.title('Elbow Method for Optimal k')
plt.show()
from sklearn.utils import resample
import numpy as np

# Define the number of bootstrap iterations
n_bootstrap = 100

# Initialize a dictionary to store bootstrap results
bootstrap_results = {}

# Set seed for reproducibility
np.random.seed(1234)

# Perform bootstrapping for each value of k
for k in range(2, 9):
    bootstrapped_clusters = []
    for _ in range(n_bootstrap):
        # Bootstrap sampling
        boot_sample = resample(MD.x, replace=True)
        # Perform k-means clustering
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(boot_sample)
        bootstrapped_clusters.append(kmeans)
    # Store the bootstrap results for this value of k
    bootstrap_results[k] = bootstrapped_clusters
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score

# Initialize lists to store results
num_segments = []
adjusted_rand_indices = []

# Calculate adjusted Rand index for each value of k and each bootstrap iteration
for k, bootstrapped_clusters in bootstrap_results.items():
    for boot_cluster in bootstrapped_clusters:
        # Predict cluster labels for the original data
        cluster_labels = boot_cluster.predict(MD.x)
        # Compute adjusted Rand index
        adjusted_rand = adjusted_rand_score(labels_true, cluster_labels) # You need to replace 'labels_true' with the true labels if available
        # Append results to lists
        num_segments.append(k)
        adjusted_rand_indices.append(adjusted_rand)

# Plot adjusted Rand index against the number of segments
plt.plot(num_segments, adjusted_rand_indices, marker='o')
plt.xlabel('Number of Segments')
plt.ylabel('Adjusted Rand Index')
plt.title('Adjusted Rand Index vs. Number of Segments')
plt.grid(True)
plt.show()
import matplotlib.pyplot as plt

# Extract cluster probabilities for cluster "4"
cluster_4_probs = MD.km28[4]

# Plot histogram
plt.hist(cluster_4_probs, bins=10, range=(0, 1))
plt.xlabel('Probability')
plt.ylabel('Frequency')
plt.title('Histogram of Cluster 4 Probabilities')
plt.xlim(0, 1)
plt.show()

# Accessing the dictionary item with the key "4" from the MD_km28 dictionary
MD_k4 = MD_km28["4"]

from flexclust import slsw_flexclust

# Assuming MD_x is your data matrix and MD_k4 contains cluster assignments
MD_r4 = slsw_flexclust(MD_x, MD_k4)

import matplotlib.pyplot as plt

# Assuming MD_r4 contains segment stability values
plt.plot(MD_r4, marker='o')  # Plot segment stability values
plt.xlabel('segment number')  # Set x-axis label
plt.ylabel('segment stability')  # Set y-axis label
plt.ylim(0, 1)  # Set y-axis limit from 0 to 1
plt.show()  # Display the plot

from flexmix import stepFlexmix, FLXMCmvbinary

# Set random seed
import numpy as np
np.random.seed(1234)

# Perform stepwise flexmix clustering
MD_m28 = stepFlexmix(MD_x, k=range(2, 9), nrep=10, model=FLXMCmvbinary(), verbose=False)

# Display the result
print(MD_m28)

import matplotlib.pyplot as plt

# Plot the information criteria (AIC, BIC, ICL)
plt.plot(MD_m28["k"], MD_m28["AIC"], label="AIC")
plt.plot(MD_m28["k"], MD_m28["BIC"], label="BIC")
plt.plot(MD_m28["k"], MD_m28["ICL"], label="ICL")
plt.xlabel("Number of Segments (k)")
plt.ylabel("Value of Information Criteria")
plt.legend()
plt.show()
from collections import Counter

# Get the clusters from kmeans and mixture models
kmeans_clusters = clusters_MD_k4
mixture_clusters = clusters_MD_m4

# Count the occurrences of each cluster combination
cluster_combinations = Counter(zip(kmeans_clusters, mixture_clusters))

# Display the cluster combinations
for kmeans_cluster, mixture_cluster in cluster_combinations:
    print(f"kmeans {kmeans_cluster} vs. mixture {mixture_cluster}: {cluster_combinations[(kmeans_cluster, mixture_cluster)]}")
from sklearn.mixture import GaussianMixture

# Fit Gaussian Mixture Model
gmm = GaussianMixture(n_components=4, covariance_type='full', random_state=1234)
gmm.fit(MD.x)

# Predict clusters
clusters_MD_m4a = gmm.predict(MD.x)

# Count the occurrences of each cluster combination
cluster_combinations = Counter(zip(clusters_MD_k4, clusters_MD_m4a))

# Display the cluster combinations
for kmeans_cluster, mixture_cluster in cluster_combinations:
    print(f"kmeans {kmeans_cluster} vs. mixture {mixture_cluster}: {cluster_combinations[(kmeans_cluster, mixture_cluster)]}")
# Calculate log likelihood for MD.m4a
log_likelihood_m4a = gmm.score(MD.x)

# Calculate log likelihood for MD.m4
log_likelihood_m4 = MD.m4$logLik

print(f"log Lik. MD.m4a: {log_likelihood_m4a}")
print(f"log Lik. MD.m4: {log_likelihood_m4}")
# Reverse the table of the 'Like' variable
like_table = {5: 152, 4: 71, 3: 73, 2: 59, 1: 58, 0: 169, -1: 152, -2: 187, -3: 229, -4: 160, -5: 143}

# Add a new column 'Like.n' to mcdonalds dataframe
mcdonalds['Like.n'] = 6 - mcdonalds['Like']

# Display the table of 'Like.n'
print(mcdonalds['Like.n'].value_counts())
# Create the formula for the regression model
columns = "+".join(mcdonalds.columns[:11])
formula = "Like.n ~ " + columns
formula
# Import necessary library
from flexmix import stepFlexmix

# Set seed for reproducibility
import numpy as np
np.random.seed(1234)

# Perform stepwise flexmix regression
MD_reg2 = stepFlexmix(f, data = mcdonalds, k = 2, nrep = 10, verbose = False)
print(MD_reg2)
# Refit the flexmix model
MD_ref2 = MD_reg2.refit()

# Summary of the refitted model
print(MD_ref2.summary())

# Plot the refitted flexmix model with significance indicators
MD_ref2.plot(significance=True)
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist

# Calculate pairwise distance matrix
distance_matrix = pdist(MD_x.T)

# Perform hierarchical clustering
MD_vclust = linkage(distance_matrix, method='single')  # Adjust method as needed

# Plot the dendrogram
dendrogram(MD_vclust)
import matplotlib.pyplot as plt

# Define colors for shading
colors = ['skyblue' if val == 1 else 'lightgray' for val in MD_k4]

# Create bar chart
plt.bar(range(len(MD_k4)), MD_k4, color=colors)

# Reverse the order based on hierarchical clustering
order = list(reversed(MD_vclust['leaves']))
plt.xticks(range(len(MD_k4)), order)  # Update x-axis labels
plt.xlabel('Segments')
plt.ylabel('Cluster')
plt.title('Bar Chart of MD.k4 with shading')
plt.show()
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Define function for projecting data onto principal components
def project_data(data, pca_model):
    return pca_model.transform(data)

# Project data onto principal components
MD_x_pca = project_data(MD_x, pca_model)

# Plot the data
plt.scatter(MD_x_pca[:, 0], MD_x_pca[:, 1], c=MD_k4, cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('MD.k4 Projected onto PCA Components')

# Add principal axes
plt.quiver(0, 0, projAxes[0, 0], projAxes[0, 1], color='red', angles='xy', scale_units='xy', scale=1)
plt.quiver(0, 0, projAxes[1, 0], projAxes[1, 1], color='blue', angles='xy', scale_units='xy', scale=1)

plt.colorbar(label='Cluster')
plt.grid()
plt.show()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.mosaicplot import mosaic

# Define data
k4 = clusters_MD_k4
like_categories = pd.cut(mcdonalds_Like, bins=np.arange(-6, 7), labels=["I hate it!", "", "", "", "", "", "", "", "", "", "I love it!"])

# Create DataFrame
data = pd.DataFrame({'k4': k4, 'Like': like_categories})

# Plot mosaic plot
mosaic(data, ['k4', 'Like'], title='Mosaic Plot', gap=0.01)

plt.xlabel('Segment Number')
plt.show()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Assuming k4 and mcdonalds_Gender are already defined
# Create a DataFrame with cluster and gender information
data = pd.DataFrame({'Cluster': k4, 'Gender': mcdonalds_Gender})

# Compute the contingency table
contingency_table = pd.crosstab(data['Cluster'], data['Gender'])

# Plot the mosaic plot
mosaic_data = contingency_table.stack().reset_index()
mosaic_data.columns = ['Cluster', 'Gender', 'Count']

plt.figure(figsize=(8, 6))
plt.mosaic(mosaic_data, index=['Cluster', 'Gender'], gap=0.01, labelizer=lambda k: '')

plt.xlabel('Gender')
plt.ylabel('Cluster')
plt.title('Mosaic Plot of Gender by Cluster')

plt.show()
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder

# Assuming mcdonalds dataset is available
# Encode categorical variables
label_encoders = {}
for col in ['Gender', 'VisitFrequency']:
    label_encoders[col] = LabelEncoder()
    mcdonalds[col] = label_encoders[col].fit_transform(mcdonalds[col])

# Define features and target variable
X = mcdonalds[['Like.n', 'Age', 'VisitFrequency', 'Gender']]
y = (k4 == 3).astype(int)  # Converting k4 to binary for classification

# Train decision tree classifier
tree_classifier = DecisionTreeClassifier()
tree_classifier.fit(X, y)

# Plot the decision tree
plt.figure(figsize=(12, 8))
plot_tree(tree_classifier, feature_names=X.columns, class_names=['Not in cluster 3', 'In cluster 3'], filled=True)
plt.show()
# Assuming k4 and mcdonalds dataset are available

# Compute the mean visit frequency for each cluster
visit = mcdonalds.groupby(k4)['VisitFrequency'].mean()

# Print the result
print(visit)
# Assuming k4 and mcdonalds dataset are available

# Compute the mean "Like" rating for each cluster
like = mcdonalds.groupby(k4)['Like.n'].mean()

# Print the result
print(like)
# Assuming k4 and mcdonalds dataset are available

# Convert Gender to binary (0 for Male, 1 for Female)
mcdonalds['Gender_binary'] = (mcdonalds['Gender'] == 'Female').astype(int)

# Compute the mean female proportion for each cluster
female = mcdonalds.groupby(k4)['Gender_binary'].mean()

# Print the result
print(female)
import matplotlib.pyplot as plt

# Plot visit vs like with marker size proportional to female proportion
plt.scatter(visit, like, s=10 * female, alpha=0.5)

# Set x-axis and y-axis limits
plt.xlim(2, 4.5)
plt.ylim(-3, 3)

# Add cluster numbers as text labels
for i, txt in enumerate(range(1, 5)):
    plt.text(visit[i], like[i], txt, ha='center', va='center')

# Set labels and title
plt.xlabel('Visit Frequency')
plt.ylabel('Likeness')
plt.title('Cluster Characteristics')

# Show plot
plt.show()




