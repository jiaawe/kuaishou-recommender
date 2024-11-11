from collections import defaultdict
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import ndcg_score

def get_user_watch_history(data):
    """
    Args: 
        data: DataFrame of user watch history. Must contain columns 'user_id' and 'video_id'.
        
    Returns:
        A dictionary with user_id as key and a set of video_ids that the user has watched as value.
    """
    watch_history_dict = defaultdict(set)
    for user in data['user_id'].unique():
        watch_history_dict[user] = set(data[data['user_id'] == user]['video_id'])
    return watch_history_dict

def get_ground_truth(ground_truth_df, valid_users, valid_videos, user_watch_history):
    """
    Args:
        ground_truth_df: DataFrame with the ground truth watch ratios.
        videos_in_train_data: List of video_ids that are present in the training data.
        user_watch_history: Dictionary with user_id as key and a list of video_ids that the user has watched as value.
    
    Returns:
        DataFrame with the ground truth watch ratios. It only contains videos that are present in training data and that the user has not watched before.
        Users that are not in the training data are filtered out as well, as we cannot make recommendations for them.
        The dataframe is sorted by user in ascending order and watch_ratio in descending order.
    """
    ground_truth_new = pd.DataFrame(columns=['user_id', 'video_id', 'watch_ratio'])

    for user in ground_truth_df['user_id'].unique():
        if user not in valid_users:
            continue
        user_ground_truth = ground_truth_df[ground_truth_df['user_id'] == user].copy()
        user_ground_truth = user_ground_truth[~user_ground_truth['video_id'].isin(user_watch_history[user])]
        user_ground_truth = user_ground_truth[user_ground_truth['video_id'].isin(valid_videos)]

        ground_truth_new = pd.concat([ground_truth_new, user_ground_truth])

    # Sort by watch_ratio in descending order
    ground_truth_new = ground_truth_new.sort_values(by=['user_id', 'watch_ratio'], ascending=[True, False])
    return ground_truth_new

def get_user_recommendations(prediction_scores, valid_videos, user_watch_history):
    """
    Args:
        prediction_scores: DataFrame with the predicted watch_ratios.
        user_watch_history: Dictionary with user_id as key and a list of video_ids that the user has watched as value.
    
    Returns:
        DataFrame with the recommendations for a specific user. It only contains videos that the user has not watched before.
        The dataframe is sorted by user in ascending order and watch_ratio in descending order.
    """
    recommendations_list = []
    
    prediction_scores = prediction_scores[prediction_scores['video_id'].isin(valid_videos)]

    for user in tqdm(prediction_scores['user_id'].unique()):
        user_recommendations = prediction_scores[prediction_scores['user_id'] == user].copy()
        user_recommendations = user_recommendations[~user_recommendations['video_id'].isin(user_watch_history[user])]
        
        recommendations_list.append(user_recommendations)

    # Concatenate all at once
    recommendations_new = pd.concat(recommendations_list)

    # Sort by prediction in descending order
    recommendations_new = recommendations_new.sort_values(by=['user_id', 'predicted_watch_ratio'], ascending=[True, False])
    return recommendations_new

def get_top_k_for_user(k, user_id, df):
    """
    Args:
        k: The number of recommendations to return.
        user_id: The user for which to get recommendations.
        df: DataFrame containing the scores for all users, sorted by score in descending order.
    
    Returns:
        DataFrame with the top k scores.
    """
    if isinstance(df, pd.core.groupby.generic.DataFrameGroupBy):
        return df.get_group(user_id).head(k)

    return df[df['user_id'] == user_id].head(k)

def get_category_tally_at_k(recommendations, video_info):
    """
    Args:
        recommendations: DataFrame with the top k recommendations for a specific user.
        video_info: DataFrame with information about the videos.

    Returns:
        Dictionary with the category as key and the number of videos in each category as value.
    """
    tally = defaultdict(int)

    for video_id in recommendations['video_id']:
        category = video_info.loc[str(video_id)]['english_first_level_category_name']
        tally[category] += 1
    
    return tally

def get_category_ndcg_at_k(recommendations, ground_truth, video_info):
    """
    Args:
        recommendations: DataFrame with the top k video recommendations for a specific user.
        ground_truth: DataFrame with the ground truth videos for a specific user.
        video_info: DataFrame with information about the videos.

    Returns:
        NDCG score for the categories of the top k recommendations.
    """
    cat_tally_reco = get_category_tally_at_k(recommendations, video_info)
    cat_tally_gt = get_category_tally_at_k(ground_truth, video_info)

    cat_tally_reco_adjusted = {}
    for category in cat_tally_gt:
        cat_tally_reco_adjusted[category] = cat_tally_reco.get(category, 0)
        
    return ndcg_score([list(cat_tally_gt.values())], [list(cat_tally_reco_adjusted.values())])

def get_average_ndcg_at_k(k, ground_truth, recommendations, video_info, by_cluster = False):
    """
    Args:
        k: The number of recommendations to return.
        ground_truth: DataFrame with the ground truth watch ratios, sorted by descending watch_ratio.
        recommendations: DataFrame with all video recommendations, sorted by descending predicted_watch_ratio.
        video_info: DataFrame with information about the videos.
        by_cluster: Boolean indicating whether to calculate the average ndcg@k per cluster.

    Returns:
        The average category-aware NDCG@k for all users, and a dictionary with the NDCG@k per cluster.
    """
    all_ndcg_scores = []

    if by_cluster:
        cluster_scores = {}
        ground_truth_users = set(ground_truth['user_id'].unique())
        
        for cluster in sorted(recommendations['cluster'].unique()):
            cluster_recommendations = recommendations[recommendations['cluster'] == cluster]
            users_in_cluster = set(cluster_recommendations['user_id'].unique())
            
            # Filter users in both ground truth and the current cluster's recommendations
            valid_users = ground_truth_users.intersection(users_in_cluster)
            cluster_ndcg_scores = []

            for user_id in tqdm(valid_users):
                user_recommendations_top_k = get_top_k_for_user(k, user_id, cluster_recommendations)
                user_ground_truth_top_k = get_top_k_for_user(k, user_id, ground_truth)

                user_ndcg_score = get_category_ndcg_at_k(user_recommendations_top_k, user_ground_truth_top_k, video_info)

                cluster_ndcg_scores.append(user_ndcg_score)
                all_ndcg_scores.append(user_ndcg_score)

            # Store the mean NDCG score per cluster
            cluster_scores[cluster] = np.mean(cluster_ndcg_scores) if cluster_ndcg_scores else 0
    else:
        users_in_train = set(recommendations.groups.keys()) if isinstance(recommendations, pd.core.groupby.generic.DataFrameGroupBy) else set(recommendations['user_id'])
        users_in_val = set(ground_truth.groups.keys()) if isinstance(ground_truth, pd.core.groupby.generic.DataFrameGroupBy) else set(ground_truth['user_id'])

        # Filter out users found in the validation set and not found in the training set. We cannot generate recommendations for these users since they do not exist in the training data
        users_in_val = users_in_val.intersection(users_in_train)
        for user_id in tqdm(users_in_val):
            user_recommendations_top_k = get_top_k_for_user(k, user_id, recommendations)
            user_ground_truth_top_k = get_top_k_for_user(k, user_id, ground_truth)

            user_ndcg_score = get_category_ndcg_at_k(user_recommendations_top_k, user_ground_truth_top_k, video_info)

            all_ndcg_scores.append(user_ndcg_score)

        cluster_scores = None

    return np.mean(all_ndcg_scores), cluster_scores

def get_user_distinct_categories_at_k(k, user_id, recommendations, video_data):
    """
    Args:
        k: The number of recommendations to return.
        user_id: The user for which to get recommendations.
        recommendations: DataFrame containing the scores for all users, sorted by score in descending order.
        
    Returns:
        The number of distinct categories in the top k recommendations.
    """
    top_k = get_top_k_for_user(k, user_id, recommendations)
    categories = set()

    for video_id in top_k['video_id']:
        category = video_data.loc[str(video_id)]['english_first_level_category_name']
        categories.add(category)
    
    return len(categories)

def get_average_distinct_categories_at_k(k, recommendations, ground_truth, video_data, by_cluster = False):
    """
    Args:
        k: The number of recommendations to return.
        recommendations: DataFrame containing the scores for all users, sorted by score in descending order.
    
    Returns:
        The overall average number of distinct categories in the top k recommendations, and a dictionary with the average number of distinct categories per cluster.
    """
    all_distinct_categories = []

    users_in_train = set(recommendations.groups.keys()) if isinstance(recommendations, pd.core.groupby.generic.DataFrameGroupBy) else set(recommendations['user_id'])
    users_in_val = set(ground_truth.groups.keys()) if isinstance(ground_truth, pd.core.groupby.generic.DataFrameGroupBy) else set(ground_truth['user_id'])

    if by_cluster:
        cluster_distinct_categories = {}

        for cluster in sorted(recommendations['cluster'].unique()):
            cluster_distinct_categories_list = []
            
            # Get users in the current cluster
            users_in_cluster = set(recommendations[recommendations['cluster'] == cluster]['user_id'])
            valid_users = users_in_val.intersection(users_in_cluster)

            for user_id in tqdm(valid_users):
                user_distinct_categories = get_user_distinct_categories_at_k(k, user_id, recommendations, video_data)
                cluster_distinct_categories_list.append(user_distinct_categories)
                all_distinct_categories.append(user_distinct_categories)

            cluster_distinct_categories[cluster] = np.mean(cluster_distinct_categories_list) if cluster_distinct_categories_list else 0
    else:
        # Filter out users found in the validation set and not found in the training set. We cannot generate recommendations for these users since they do not exist in the training data
        users_in_val = users_in_val.intersection(users_in_train)
        for user_id in tqdm(users_in_val):
            user_distinct_categories = get_user_distinct_categories_at_k(k, user_id, recommendations, video_data)

            all_distinct_categories.append(user_distinct_categories)

        cluster_distinct_categories = None
    
    return np.mean(all_distinct_categories), cluster_distinct_categories

def get_user_avg_watch_ratio_at_k(k, user_id, recommendations, watch_ratio_column, ground_truth):
    """
    Args:
        k: The number of recommendations to return.
        user_id: The user for which to get recommendations.
        recommendations: DataFrame containing the scores for all users, sorted by score in descending order.
        
    Returns:
        The average watch_ratio in the top k recommendations.
    """
    reco_subset = recommendations.get_group(user_id) if isinstance(recommendations, pd.core.groupby.generic.DataFrameGroupBy) else recommendations[recommendations['user_id'] == user_id]
    ground_truth_subset = ground_truth.get_group(user_id) if isinstance(ground_truth, pd.core.groupby.generic.DataFrameGroupBy) else ground_truth[ground_truth['user_id'] == user_id]

    video_ids = set(ground_truth_subset['video_id'])

    top_k = set(reco_subset[reco_subset['video_id'].isin(video_ids)].head(k)['video_id'].tolist())

    return np.mean(ground_truth_subset[ground_truth_subset['video_id'].isin(top_k)][watch_ratio_column])

def get_avg_watch_ratio_at_k(k, recommendations, ground_truth, by_cluster = False):
    """
    Args:
        k: The number of recommendations to return.
        recommendations: DataFrame containing the scores for all users, sorted by score in descending order.
        by_cluster: Boolean indicating whether to calculate the average watch_ratio per cluster.
        
    Returns:
        The overall average watch_ratio in the top k recommendations, and a dictionary with the average predicted_watch_ratio per cluster (if by_cluster is True).
    """
    all_avg_watch_ratios_list = []

    users_in_val = set(ground_truth.groups.keys()) if isinstance(ground_truth, pd.core.groupby.generic.DataFrameGroupBy) else set(ground_truth['user_id'])
    
    cluster_avg_watch_ratios = {}

    if by_cluster:
        for cluster in sorted(recommendations['cluster'].unique()):
            cluster_avg_watch_ratios_list = []

            # Get users in the current cluster and intersect with validation users
            users_in_cluster = set(recommendations[recommendations['cluster'] == cluster]['user_id'])
            valid_users = users_in_val.intersection(users_in_cluster)

            for user_id in tqdm(valid_users):
                user_avg_watch_ratio = get_user_avg_watch_ratio_at_k(k, user_id, recommendations, 'watch_ratio', ground_truth)

                cluster_avg_watch_ratios_list.append(user_avg_watch_ratio)
                all_avg_watch_ratios_list.append(user_avg_watch_ratio)
            
            cluster_avg_watch_ratios[cluster] = np.mean(cluster_avg_watch_ratios_list) if cluster_avg_watch_ratios_list else 0
    else:
        users_in_train = set(recommendations.groups.keys()) if isinstance(recommendations, pd.core.groupby.generic.DataFrameGroupBy) else set(recommendations['user_id'])

        # Filter out users found in the validation set and not found in the training set. We cannot generate recommendations for these users since they do not exist in the training data
        users_in_val = users_in_val.intersection(users_in_train)
        for user_id in tqdm(users_in_val):
            user_avg_watch_ratio = get_user_avg_watch_ratio_at_k(k, user_id, recommendations, 'watch_ratio', ground_truth)

            all_avg_watch_ratios_list.append(user_avg_watch_ratio)
    
    return np.mean(all_avg_watch_ratios_list), cluster_avg_watch_ratios

def get_user_precision_recall_f1_at_k(k, user_id, recommendations, ground_truth, threshold):
    """
    Args:
        k: The number of recommendations to return.
        user_id: The user for which to get recommendations.
        recommendations: DataFrame containing the scores for all users, sorted by score in descending order.
        ground_truth: DataFrame with the ground truth watch ratios.
        threshold: The threshold for the watch ratio.
    
    Returns:
        Precision, recall, and F1 score at k for a specific user.
    """
    reco_subset = recommendations.get_group(user_id) if isinstance(recommendations, pd.core.groupby.generic.DataFrameGroupBy) else recommendations[recommendations['user_id'] == user_id]
    ground_truth_subset = ground_truth.get_group(user_id) if isinstance(ground_truth, pd.core.groupby.generic.DataFrameGroupBy) else ground_truth[ground_truth['user_id'] == user_id]

    video_ids = set(ground_truth_subset['video_id'])

    reco_subset = reco_subset[reco_subset['video_id'].isin(video_ids)].head(k)
    
    tp = 0
    fp = 0

    for video_id in reco_subset['video_id']:
        if video_id in ground_truth_subset['video_id'].values:
            if ground_truth_subset[ground_truth_subset['video_id'] == video_id]['watch_ratio'].values[0] >= threshold:
                # If the video is in top_k_ground_truth and watch ratio is above the threshold, it is a true positive
                tp += 1
            else:
                # If the video is in top_k_ground_truth but watch ratio is below the threshold, it is a false positive
                fp += 1
                
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall  = tp / np.sum(ground_truth_subset['watch_ratio'] >= threshold)
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    return precision, recall, f1

def get_precision_recall_f1_at_k(k, recommendations, ground_truth, threshold, by_cluster = False):
    """
    Args:
        k: The number of recommendations to return.
        recommendations: DataFrame containing the scores for all users, sorted by score in descending order.
        ground_truth: DataFrame with the ground truth watch ratios.
        threshold: The threshold for the watch ratio.
        by_cluster: Boolean indicating whether to calculate the precision, recall, and F1 score per cluster.
    
    Returns:
        The overall average precision, recall, and F1 score at k, and a dictionary with the average precision, recall, and F1 score per cluster if by_cluster is True.
    """
    all_precision_list = []
    all_recall_list = []
    all_f1_list = []

    if by_cluster:
        cluster_precision = {}
        cluster_recall = {}
        cluster_f1 = {}
        users_in_val = set(ground_truth['user_id'])

        for cluster in sorted(recommendations['cluster'].unique()):
            cluster_precision_list = []
            cluster_recall_list = []
            cluster_f1_list = []

            # Get users in the current cluster and intersect with validation users
            users_in_cluster = set(recommendations[recommendations['cluster'] == cluster]['user_id'])
            valid_users = users_in_val.intersection(users_in_cluster)

            for user_id in tqdm(valid_users):
                user_precision, user_recall, user_f1 = get_user_precision_recall_f1_at_k(
                    k, user_id, recommendations, ground_truth, threshold
                )

                cluster_precision_list.append(user_precision)
                cluster_recall_list.append(user_recall)
                cluster_f1_list.append(user_f1)

                all_precision_list.append(user_precision)
                all_recall_list.append(user_recall)
                all_f1_list.append(user_f1)
            
            # Calculate cluster-specific averages
            cluster_precision[cluster] = np.mean(cluster_precision_list) if cluster_precision_list else 0
            cluster_recall[cluster] = np.mean(cluster_recall_list) if cluster_recall_list else 0
            cluster_f1[cluster] = np.mean(cluster_f1_list) if cluster_f1_list else 0
    else:
        users_in_train = set(recommendations.groups.keys()) if isinstance(recommendations, pd.core.groupby.generic.DataFrameGroupBy) else set(recommendations['user_id'])
        users_in_val = set(ground_truth.groups.keys()) if isinstance(ground_truth, pd.core.groupby.generic.DataFrameGroupBy) else set(ground_truth['user_id'])

        # Filter out users found in the validation set and not found in the training set. We cannot generate recommendations for these users since they do not exist in the training data
        users_in_val = users_in_val.intersection(users_in_train)

        for user_id in tqdm(users_in_val):
            user_precision, user_recall, user_f1 = get_user_precision_recall_f1_at_k(k, user_id, recommendations, ground_truth, threshold)

            all_precision_list.append(user_precision)
            all_recall_list.append(user_recall)
            all_f1_list.append(user_f1)

        cluster_precision = None
        cluster_recall = None
        cluster_f1 = None
    
    return (
        np.mean(all_precision_list),
        np.mean(all_recall_list),
        np.mean(all_f1_list),
        cluster_precision,
        cluster_recall,
        cluster_f1
    )

def get_all_metrics(k1, k2, ground_truth, recommendations, video_info, threshold, by_cluster = False):
    """
    Args:
        k1: The number of recommendations to return for NDCG@k, distinct categories, and avg watch ratio.
        k2: The number of recommendations to return for precision, recall, and F1 score.
        ground_truth: DataFrame with the ground truth watch ratios, sorted by descending watch_ratio.
        recommendations: DataFrame with all video recommendations, sorted by descending predicted_watch_ratio.
        video_info: DataFrame with information about the videos.
        threshold: The threshold for the watch ratio to calculate binary labels.
        by_cluster: Boolean indicating whether to calculate the metrics per cluster.

    Returns:
        Dataframe of all evaluation metrics.
    """
    overall_ndcg, cluster_ndcg = get_average_ndcg_at_k(k1, ground_truth, recommendations, video_info, by_cluster)
    overall_distinct_categories, cluster_distinct_categories = get_average_distinct_categories_at_k(k1, recommendations, ground_truth, video_info, by_cluster)
    overall_avg_watch_ratio, cluster_avg_watch_ratio = get_avg_watch_ratio_at_k(k1, recommendations, ground_truth, by_cluster)
    avg_precision, avg_recall, avg_f1, cluster_precision, cluster_recall, cluster_f1 = get_precision_recall_f1_at_k(k2, recommendations, ground_truth, threshold, by_cluster)

    metrics_df = pd.DataFrame(columns=['cluster', f'NDCG@{k1}', f'Distinct Categories @ {k1}', f'Avg Watch Ratio @ {k1}', f'Avg Precision@{k2}', f'Avg Recall@{k2}', f'Avg F1@{k2}'])
    if by_cluster:
        for cluster in recommendations['cluster'].unique():
            metrics_df = pd.concat([
                metrics_df,
                pd.DataFrame({
                    'cluster': cluster,
                    f'NDCG@{k1}': cluster_ndcg[cluster],
                    f'Distinct Categories @ {k1}': cluster_distinct_categories[cluster],
                    f'Avg Watch Ratio @ {k1}': cluster_avg_watch_ratio[cluster],
                    f'Avg Precision@{k2}': cluster_precision[cluster],
                    f'Avg Recall@{k2}': cluster_recall[cluster],
                    f'Avg F1@{k2}': cluster_f1[cluster]
                }, index=[0])
            ])
    metrics_df['cluster'] = metrics_df['cluster'].astype(int)
    metrics_df = metrics_df.sort_values(by='cluster')
    
    metrics_df = pd.concat([
        metrics_df,
        pd.DataFrame({
            'cluster': 'Overall',
            f'NDCG@{k1}': overall_ndcg,
            f'Distinct Categories @ {k1}': overall_distinct_categories,
            f'Avg Watch Ratio @ {k1}': overall_avg_watch_ratio,
            f'Avg Precision@{k2}': avg_precision,
            f'Avg Recall@{k2}': avg_recall,
            f'Avg F1@{k2}': avg_f1
        }, index=[0])
    ])

    return metrics_df