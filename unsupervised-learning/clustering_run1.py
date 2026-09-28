import numpy as np
from k_means import KMeans
from EM import EMGMM
import matplotlib.pyplot as plt


def generate_dataset_with_source_info(N, S, sigma, spacing_factor):
    """
    Generate dataset from S Gaussian sources, and also return source information.
    """
    np.random.seed(42)  # for reproducibility
    total_data = np.empty((0, 2))
    source_labels = []
    source_means = []
    source_std_devs = []
    spacing = np.cumsum(np.full(S, spacing_factor * sigma))  # Cumulative sum to increase spacing

    for s in range(S):
        mean = np.array([spacing[s], 0])  # Increase spacing horizontally
        std_dev = sigma  # Standard deviation is constant in this setup
        source_data = np.random.normal(loc=mean, scale=std_dev, size=(N, 2))
        total_data = np.vstack((total_data, source_data))
        source_labels.extend([s] * N)  # Track source label for each point
        source_means.append(mean)
        source_std_devs.append(std_dev)

    return total_data, np.array(source_labels), np.array(source_means), np.array(source_std_devs)

def create_datasets_with_info(N=1000, S_values=[3, 5, 10], sigma=1, spacing_factors=[0.5, 1, 1.5, 2]):
    datasets = {}
    source_info = {}
    for S in S_values:
        for spacing in spacing_factors:
            key = f"S={S}_spacing={spacing}σ"
            data, labels, means, std_devs = generate_dataset_with_source_info(N, S, sigma, spacing)
            datasets[key] = data
            source_info[key] = {'source_labels': labels, 'means': means, 'std_devs': std_devs}
    return datasets, source_info


def run_clustering_experiments(datasets, S_values=[3, 5, 10], cluster_configs={3: [2, 3, 6, 8], 5: [5], 10: [10]}):
    results = {}
    for key, data in datasets.items():
        S = int(key.split('_')[0].split('=')[1])  # Extract the number of sources from the key
        for n_clusters in cluster_configs[S]:
            kmeans = KMeans(n_clusters=n_clusters)
            kmeans.fit(data)
            em = EMGMM(n_components=n_clusters)
            em.fit(data)

            results_key = f"{key}_clusters={n_clusters}"
            results[results_key] = {
                "kmeans": {"labels": kmeans.predict(data), "centers": kmeans.centers},
                "em": {"labels": em.predict(data), "means": em.means}
            }
    return results

# Create datasets along with source information
datasets, source_info = create_datasets_with_info()

# Run experiments
results = run_clustering_experiments(datasets)

# Clustering visualization
def visualize_clustering(data, labels, centers, title):
    """
    Visualize the clustering output.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(data[:, 0], data[:, 1], c=labels, alpha=0.5, edgecolors='k', cmap='viridis')
    plt.scatter(centers[:, 0], centers[:, 1], c='red', s=100, marker='x')  # Mark cluster centers
    plt.title(title)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.colorbar(label='Cluster ID')
    plt.grid(True)
    plt.show()

def plot_all_results(datasets, results):
    """
    Plot results for all datasets and conditions.
    """
    for key, value in results.items():
        data = datasets[key.rsplit('_', 1)[0]]  # Extract dataset key from results key
        # KMeans results
        kmeans_labels = value['kmeans']['labels']
        kmeans_centers = value['kmeans']['centers']
        em_labels = value['em']['labels']
        em_means = value['em']['means']

        # Visualize KMeans
        visualize_clustering(data, kmeans_labels, kmeans_centers, f'KMeans Clustering - {key}')
        # Visualize EM
        visualize_clustering(data, em_labels, em_means, f'EM Clustering - {key}')

# Example call to plot results
plot_all_results(datasets, results)


# Evaluation
def evaluate_homogeneity(data, labels, source_labels):
    """
    Calculate homogeneity of clusters, i.e., the fraction of pairs belonging to the same source.
    """
    cluster_ids = np.unique(labels)
    homogeneity_scores = []

    for cluster_id in cluster_ids:
        cluster_mask = (labels == cluster_id)
        cluster_sources = source_labels[cluster_mask]
        if len(cluster_sources) > 1:
            most_common = np.bincount(cluster_sources).max()
            homogeneity = most_common / cluster_sources.size
            homogeneity_scores.append(homogeneity)
        else:
            homogeneity_scores.append(1)  # Perfect homogeneity if the cluster has a single element

    if len(homogeneity_scores) == 0:
        return 0

    return np.mean(homogeneity_scores)

def compare_cluster_parameters(data, labels, source_labels, source_means, source_std_devs):
    """
    Compare the estimated cluster parameters to the actual source distribution parameters.
    """
    cluster_ids = np.unique(labels)
    parameter_differences = []

    for cluster_id in cluster_ids:
        cluster_mask = (labels == cluster_id)
        cluster_data = data[cluster_mask]
        cluster_source_labels = source_labels[cluster_mask]

        if cluster_data.size == 0:
            continue

        # Determine the most common source in the cluster
        most_common_source = np.bincount(cluster_source_labels).argmax()
        mean_diff = np.linalg.norm(np.mean(cluster_data, axis=0) - source_means[most_common_source])
        std_dev_diff = np.linalg.norm(np.std(cluster_data, axis=0) - source_std_devs[most_common_source])

        parameter_differences.append((mean_diff, std_dev_diff))

    return parameter_differences

def run_and_evaluate_experiments(datasets, source_info):
    results = run_clustering_experiments(datasets)
    evaluation_results = {}

    for key, value in results.items():
        data = datasets[key.rsplit('_', 1)[0]]
        source_labels = source_info[key.rsplit('_', 1)[0]]['source_labels']
        source_means = source_info[key.rsplit('_', 1)[0]]['means']
        source_std_devs = source_info[key.rsplit('_', 1)[0]]['std_devs']

        # KMeans evaluation
        kmeans_labels = value['kmeans']['labels']
        kmeans_homogeneity = evaluate_homogeneity(data, kmeans_labels, source_labels)
        kmeans_parameter_diffs = compare_cluster_parameters(data, kmeans_labels, source_labels, source_means, source_std_devs)

        # EM evaluation
        em_labels = value['em']['labels']
        em_homogeneity = evaluate_homogeneity(data, em_labels, source_labels)
        em_parameter_diffs = compare_cluster_parameters(data, em_labels, source_labels, source_means, source_std_devs)

        evaluation_results[key] = {
            'kmeans': {
                'homogeneity': kmeans_homogeneity,
                'parameter_differences': kmeans_parameter_diffs
            },
            'em': {
                'homogeneity': em_homogeneity,
                'parameter_differences': em_parameter_diffs
            }
        }

    return evaluation_results

# Run experiments and evaluate
evaluation_results = run_and_evaluate_experiments(datasets, source_info)

def save_evaluation_results_to_txt(evaluations, filename="evaluation_results.txt"):
    """
    Save the evaluation metrics to a text file.
    """
    with open(filename, "w") as file:
        for key, eval_data in evaluations.items():
            kmeans_eval = eval_data['kmeans']
            em_eval = eval_data['em']
            
            file.write(f"Results for {key}:\n")
            file.write(f"  KMeans Clustering:\n")
            file.write(f"    Homogeneity: {kmeans_eval['homogeneity']:.2f}\n")
            file.write(f"    Parameter Differences:\n")
            for i, diff in enumerate(kmeans_eval['parameter_differences']):
                file.write(f"      Cluster {i+1} - Mean Difference: {diff[0]:.2f}, Standard Deviation Difference: {diff[1]:.2f}\n")

            file.write(f"  EM Clustering:\n")
            file.write(f"    Homogeneity: {em_eval['homogeneity']:.2f}\n")
            file.write(f"    Parameter Differences:\n")
            for i, diff in enumerate(em_eval['parameter_differences']):
                file.write(f"      Cluster {i+1} - Mean Difference: {diff[0]:.2f}, Standard Deviation Difference: {diff[1]:.2f}\n")
            file.write("\n")

# Assuming `evaluation_results` is your dictionary containing all the metrics
save_evaluation_results_to_txt(evaluation_results, "my_clustering_evaluation_results.txt")
