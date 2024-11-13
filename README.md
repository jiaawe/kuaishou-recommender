# Kauishou-Recommender
[Detailed Project Report](https://docs.google.com/document/d/1OpWq5wbBZ8pNEb03pQTqCD5jQbvm4Bb3A9tuFwu4y5o/edit?tab=t.0#heading=h.xrhzr61o6zn) <br>
[Dataset Link](https://kuairec.com/) <br><br>

## Abstract
Kuaishou, the first short video platform in China, operates in a highly competitive industry with many platforms striving to capture user attention and loyalty. To maintain its leading position, Kuaishou relies on effective recommendation algorithms to personalise user experiences by suggesting content that aligns with individual preferences and viewing habits. This enhances user engagement and retention, as viewers are more likely to stay on the platform when presented with videos that closely match their interests. However, challenges such as rapidly changing user preferences and the vast amount of content available can make it difficult to provide consistently relevant recommendations. Our project addresses these challenges by improving recommendation relevance and personalization.

We used the KuaiRec dataset, which holds real-world recommendation logs from the Kuaishou mobile app (Gao et al., 2022). This dataset captures a diverse range of user interactions and video content, allowing us to model and predict user engagement more effectively. By focusing on predicting the watch_ratio, which indicates whether users spend more time watching recommended videos, we align our model with important business goals like increasing revenue, session durations, and daily active users. Our approach involves analysing user behaviour patterns and their interactions with video features to improve the relevance of recommendations. Ultimately, our project aims to increase user engagement and retention, helping Kuaishou succeed in the fast-changing world of video-sharing applications.


## Project Pipeline
The following steps highlight our project's pipeline, with each file being named appropriately, from step 0 to step 8.

### Step 0: Data Collection and Translation
- Downloaded the KuaiRec dataset containing user interactions, video metadata and all other csv files
- Translated Chinese video captions and descriptions to English for better analysis (using open-source Qwen2.5)
- To download the dataset, just run step0_download_and_translate.ipynb

### Step 1: Data Preprocessing
- Cleaned and standardized raw data
- Handled missing values and outliers
- Concatenated relevant information user features and user-item informations into joined tables 
- Initial feature filtering and selection techniques
- Created initial dataset splits for training, validation and testing

### Step 2: Exploratory Data Analysis (EDA)
- Analyzed user behavior patterns
- Investigated video popularity distributions
- Examined watch time patterns
- Identified key trends in user engagement
- Visualized important relationships between variables

### Step 3: Feature Engineering
- Created user/video/user-video interaction features
- Developed additional user content preferences and user watching behaviours (such as processing temporal features to identify weekend/time-based viewing behaviours)
- New user identification, social engagement features and content creator identification
- Generated content-based features from video metadata
- Preprocessed video 'scaptions and categories and constructed video embeddings from translated caption and categories

### Step 4: Customer Segmentation
- Selected user features for clustering to identify customer segments
- Engineered additional features based on the top favourite categories (based on watch ratio) for each user
- Implementation of PCA, Elbow-method and clustering algorithms to identify user segments 
- Analysed viewing patterns within segments and user profiles based on content preferences
- Explored segmentation insights for targeted recommendations


### Step 5: Model Development
We implemented two different approaches for recommendation, analysing each model's evaluation metrics before finally performing hybridisation:

#### 5.1 Neural Collaborative Filtering (NCF)
- Implemented NCF architecture combining generalized matrix factorization and multi-layer perceptron
- Integrated time decay for our recommendations, based on the age of the videos
- Optimized model hyperparameters based on a separate set of validation data

#### 5.2 Caption-Based Model
- Utilized video caption data for content-based recommendations
- Implemented text embedding techniques
- Created similarity metrics based on video content
- Weighted with watch ratios of user's watch history, and also integrated time decay for recommendations based on the age of videos

#### 5.3 Random Baseline Model
- Developed baseline model using random recommendations
- Randomly predict watch ratios, rank them, then recommend top K videos

### Step 6: Regression Analysis for Hybridisation
Regression analysis with the validation dataset to:
1) Identify whether each model is statistically significant
2) Obtain the weights/coefficients for each model in the hybridisation step

### Step 7: Hybrid Model Development
Combined the two model approaches (caption-based recommendations and user-based NCF) via weights obtained from regression analysis

### Step 8: Model Evaluation
Implemented comprehensive evaluation metrics: 
- Average Watch Ratio
- Precision@K
- Recall@K
- F1-Score@K
- Category-Aware NDCG
- Category Diversity

<br>

These models are all being evaluated: 
- Baseline
- NCF ONLY (with segmentation)
- NCF ONLY (without segmentation)
- Caption-based model ONLY
- Hybrid model

## Other files & folders
- eval_fns.py contains all evaluation metrics, abstracted out for cleaner notebook codes
- translation_folder contains files used for testing and running LLMs
- eda folder contains previous files used in mid-term report for EDA of the KuaiRec dataset
- feature_engineering folder contains previous files used in mid-term report for initial preprocessing and feature engineering/eda of the Kuairec dataset

## References
Gao, C., Wang, Y., Ma, X., Wang, M., Feng, X., Jiang, Q., & Yang, X. (2022). KuaiRec: A Fully-observed Dataset and Insights for Evaluating Recommender Systems. arXiv preprint arXiv:2202.10842.
