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
    spacing = np.cumsum(np.full(S, spacing_factor))  # Fixed spacing

    for s in range(S):
        mean = np.array([spacing[s], 0])  # Increase spacing horizontally
        std_dev = sigma
        source_data = np.random.normal(loc=mean, scale=std_dev, size=(N, 2))
        total_data = np.vstack((total_data, source_data))
        source_labels.extend([s] * N)
        source_means.append(mean)
        source_std_devs.append(std_dev)

    return total_data, np.array(source_labels), np.array(source_means), np.array(source_std_devs)

def create_datasets_with_info(N_values=[100, 1000, 5000], S=5, spacing=1.25, sigma=0.75):
    datasets = {}
    source_info = {}
    for N in N_values:
        key = f"S={S}_spacing={spacing}_sigma={sigma}_N={N}"
        data, labels, means, std_devs = generate_dataset_with_source_info(N, S, sigma, spacing)
        datasets[key] = data
        source_info[key] = {'source_labels': labels, 'means': means, 'std_devs': std_devs}
    return datasets, source_info

def run_clustering_experiments(datasets, cluster_config=5):
    results = {}
    for key, data in datasets.items():
        kmeans = KMeans(n_clusters=cluster_config)
        kmeans.fit(data)
        em = EMGMM(n_components=cluster_config)
        em.fit(data)

        results_key = key + f"_clusters={cluster_config}"
        results[results_key] = {
            "kmeans": {"labels": kmeans.predict(data), "centers": kmeans.centers},
            "em": {"labels": em.predict(data), "means": em.means}
        }
    return results

datasets, source_info = create_datasets_with_info()

results = run_clustering_experiments(datasets)

def visualize_clustering(data, labels, centers, title):
    plt.figure(figsize=(10, 6))
    plt.scatter(data[:, 0], data[:, 1], c=labels, alpha=0.5, edgecolors='k', cmap='viridis')
    plt.scatter(centers[:, 0], centers[:, 1], c='red', s=100, marker='x')
    plt.title(title)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.colorbar(label='Cluster ID')
    plt.grid(True)
    plt.show()

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
        data = datasets[key.split("_clusters=")[0]]
        source_labels = source_info[key.split("_clusters=")[0]]['source_labels']
        source_means = source_info[key.split("_clusters=")[0]]['means']
        source_std_devs = source_info[key.split("_clusters=")[0]]['std_devs']

        # KMeans evaluation and visualization
        kmeans_labels = value['kmeans']['labels']
        kmeans_centers = value['kmeans']['centers']
        kmeans_homogeneity = evaluate_homogeneity(data, kmeans_labels, source_labels)
        kmeans_parameter_diffs = compare_cluster_parameters(data, kmeans_labels, source_labels, source_means, source_std_devs)
        
        visualize_clustering(data, kmeans_labels, kmeans_centers, f'KMeans Clustering - {key}')

        # EM evaluation and visualization
        em_labels = value['em']['labels']
        em_means = value['em']['means']
        em_homogeneity = evaluate_homogeneity(data, em_labels, source_labels)
        em_parameter_diffs = compare_cluster_parameters(data, em_labels, source_labels, source_means, source_std_devs)
        
        visualize_clustering(data, em_labels, em_means, f'EM Clustering - {key}')

        evaluation_results[key] = {
            'kmeans': {'homogeneity': kmeans_homogeneity, 'parameter_differences': kmeans_parameter_diffs},
            'em': {'homogeneity': em_homogeneity, 'parameter_differences': em_parameter_diffs}
        }

    return evaluation_results

# Now run experiments and evaluate
evaluation_results = run_and_evaluate_experiments(datasets, source_info)


def save_evaluation_results_to_txt(evaluation_results, filename="evaluation_results.txt"):
    with open(filename, "w") as file:
        for key, eval_data in evaluation_results.items():
            file.write(f"Results for {key}:\n")
            file.write("KMeans Clustering:\n")
            file.write(f"  Homogeneity: {eval_data['kmeans']['homogeneity']:.2f}\n")
            file.write("  Parameter Differences:\n")
            for diff in eval_data['kmeans']['parameter_differences']:
                file.write(f"    Mean Difference: {diff[0]:.2f}, Std Dev Difference: {diff[1]:.2f}\n")
            
            file.write("EM Clustering:\n")
            file.write(f"  Homogeneity: {eval_data['em']['homogeneity']:.2f}\n")
            file.write("  Parameter Differences:\n")
            for diff in eval_data['em']['parameter_differences']:
                file.write(f"    Mean Difference: {diff[0]:.2f}, Std Dev Difference: {diff[1]:.2f}\n")
            file.write("\n")

# Make sure to call this function with the correct data
evaluation_results = run_and_evaluate_experiments(datasets, source_info)
save_evaluation_results_to_txt(evaluation_results)
