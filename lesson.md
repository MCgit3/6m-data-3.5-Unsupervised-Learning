# Unsupervised Learning: Discovering Hidden Patterns

## Why This Matters: The Supermarket Analogy

Imagine a supermarket manager wants to rearrange shelves to improve customer experience. Instead of asking customers "which items do you buy together?", the manager watches buying behaviour over months and notices patterns: people who buy pasta also buy tomato sauce; parents with young children often buy nappies and wipes together; late-night shoppers tend to grab ready-made meals.

**No one told the system the 'right' groups—it discovered them from behaviour patterns alone.**

This is unsupervised learning. You have data (customer purchases) but no labels (which items *should* go together). Your job is to find hidden structure: groups of similar items, unusual purchases that don't fit normal patterns, and underlying themes in the data.

In this lesson, you'll learn three core unsupervised techniques:
1. **Outlier Detection** — finding the unusual data points
2. **Dimensionality Reduction** — simplifying complex data while keeping what matters
3. **Clustering** — grouping similar data points together

---

## Part 1: Outlier Detection (40 minutes)

### Why Outliers Matter

Outliers are data points that don't fit the typical pattern. They're important because:

- **They corrupt models**: A supervised learning model trained on data with outliers learns false patterns. One fraud transaction can mislead a credit model.
- **They signal real problems**: A spike in website response time might indicate a server attack. A sudden drop in patient heart rate might indicate a medical emergency.
- **They hide in plain sight**: In a dataset of 10,000 normal transactions, 5 fraudulent ones might go unnoticed—until they're found.

### Statistical Methods: IQR and Z-Score

**Interquartile Range (IQR)**

The IQR captures the "middle 50%" of your data. Any point beyond the whiskers (typically 1.5 × IQR above Q3 or below Q1) is flagged as an outlier.

*Real example*: In credit card transaction amounts:
- Q1 (25th percentile): $50
- Q3 (75th percentile): $200
- IQR: $150
- Upper whisker: $200 + 1.5 × $150 = $425
- A transaction for $5,000 is far beyond $425 → **outlier**

**Z-Score**

The z-score measures how many standard deviations a point is from the mean. Points with |z-score| > 3 are typically considered outliers.

*When to use IQR vs Z-Score*:
- **IQR**: Works well when data has a few extreme outliers but you don't assume a specific distribution (e.g., income data often has long right tails)
- **Z-Score**: Works well when data is roughly normally distributed and low-dimensional; fails if you have extreme values that inflate the standard deviation

### Machine Learning Method: Isolation Forest

**The Core Idea**: Unusual points are easier to isolate. Imagine a room with 100 people: it takes many questions ("Are you over 6 feet tall?", "Do you speak French?") to isolate a specific person, but only one or two questions to isolate an unusual person (e.g., "Are you a professional athlete?" might isolate just one).

**How It Works**:
1. Randomly split the data using random thresholds on random features
2. Points that get isolated quickly (require fewer splits) are likely outliers
3. Points in dense regions require many splits to isolate
4. The algorithm is "anomaly-aware": it naturally finds rare points

**Why Isolation Forest Wins**:
- Works in high dimensions (z-score struggles when you have 50 features)
- Doesn't assume a distribution (unlike z-score)
- Handles mixed-type outliers (one unusual feature, or many mildly unusual features)

**Key Parameter**: `contamination` — the proportion of outliers you expect (e.g., 0.05 = 5% of data is anomalous). Set this based on domain knowledge, not just "guess".

### Quick Check: Outlier Detection

**Question 1**: You have a dataset of customer purchase amounts. The mean is $100 and the standard deviation is $20. A purchase of $160 has a z-score of 3.0. Should you flag this as an outlier? Why or why not?

*Sample Answer*: Yes, you should flag it. A z-score of 3.0 means the purchase is 3 standard deviations from the mean, which is rare (happens in ~0.3% of normally distributed data). However, you should investigate before removing it—it might be a bulk purchase (legitimate) rather than fraud.

**Question 2**: You're using Isolation Forest to detect fraudulent bank transactions. Your `contamination` parameter is 0.01 (expecting 1% fraud). After training, the model flags 150 transactions in a dataset of 15,000 as anomalies. Is this expected? What might it mean?

*Sample Answer*: No, it's not as expected. You'd expect 0.01 × 15,000 = 150 anomalies, so this is actually *exactly* expected (the contamination parameter controls how many points are labeled anomalies). However, you should verify that these 150 points match your true fraud rate—if your actual fraud rate is 0.5%, the model is over-flagging.

**Question 3**: When would you prefer Isolation Forest over IQR for outlier detection?

*Sample Answer*: When your data is high-dimensional (many features) or when you don't know the distribution shape. IQR is simple and fast but only looks at one feature at a time; Isolation Forest considers all features together. Example: detecting unusual customers based on 20 behavioural features (login frequency, purchase amount, account age, etc.)—Isolation Forest would find someone with unusual *combinations* of features, not just one extreme value.

---

## Part 2: Dimensionality Reduction with PCA (30 minutes)

### The Curse of Dimensionality

More features should help, right? Not always.

**The Problem**: With too many features:
- Models need more training data to learn reliably
- Computation becomes slower
- Visualization becomes impossible (you can't plot 50 dimensions)
- Noise is amplified (more features = more chances for random noise)
- Many features are correlated—they're redundant

Imagine trying to describe a person: height, weight, shoe size, hand width, finger length, arm span, leg length. Many of these are highly correlated (tall people are generally heavier and have larger feet). You're carrying redundant information.

### PCA: Compression Without Loss of Essence

**Principal Component Analysis (PCA)** finds new axes that capture the most variance in your data.

**The Analogy**: Compressing a photograph. A full-resolution photo has millions of pixels. JPEG compression throws away information you won't notice (subtle colour gradations in uniform areas) but keeps edges and sharp details. You get a smaller file (fewer features) but the image still looks recognizable.

**How PCA Works**:
1. Find the direction of maximum variance in the data (1st principal component)
2. Find the direction of second-most variance, perpendicular to the first (2nd principal component)
3. Continue until you've captured enough variance

**Example**: Customer survey data with 50 questions. PCA might find:
- **PC1** (40% of variance): correlates strongly with product satisfaction questions—this captures "customer happiness"
- **PC2** (20% of variance): correlates with price/value questions—this captures "price sensitivity"
- Together, PC1 and PC2 explain 60% of the total variation in customer opinions

Instead of analyzing 50 features, you now analyze 2 components. You've reduced noise and kept what matters.

### Explained Variance: How Many Components Do You Need?

After PCA, each component has an "explained variance" score (percentage of total variation it captures).

```
Component 1: 32% of variance
Component 2: 18% of variance
Component 3: 12% of variance
Component 4: 08% of variance
...
```

**Rule of thumb**: Keep enough components to explain 80-95% of variance. Here, components 1-4 explain 70%, so you might add component 5 to reach 80%.

Why 80-95%? 
- Below 80%: you're losing too much information
- Above 95%: you're keeping noise and defeating the purpose of reduction

### Quick Check: PCA

**Question 1**: You apply PCA to 30 housing features and find that the first 5 components explain 85% of variance. Should you keep all 5 components or try to use fewer? Explain your reasoning.

*Sample Answer*: Keep all 5. At 85% explained variance, you're in the sweet spot—you've compressed 30 dimensions down to 5 while retaining most signal. If you dropped component 5, you'd be down to 82% (losing information), which might not be worth the extra simplicity. But if explained variance of PC1-4 was 80%, you'd drop PC5 since you already reached your threshold.

**Question 2**: In a machine learning model, you replace 20 original features with 4 PCA components. Training time decreases dramatically, but prediction accuracy on the test set also drops slightly. Is this good or bad? What does it tell you?

*Sample Answer*: This is normal and often good. The dropped accuracy reflects lost noise, not lost signal. Fewer features reduce overfitting and training time. However, investigate whether the accuracy loss is acceptable for your use case. If prediction accuracy was 95% and dropped to 94%, that's likely fine. If it dropped to 70%, the original features were necessary. Also, PCA components are harder to interpret (you can't easily explain what they mean to a business stakeholder), so trade off interpretability for simplicity.

---

## Part 3: Clustering (70 minutes)

### The Goal of Clustering

Clustering groups similar data points together—but without any labels telling you what "similar" means. You discover natural groupings from the data alone.

**Real-world uses**:
- E-commerce: customer segments for targeted marketing
- Biology: grouping genes with similar expression patterns
- Geography: identifying urban sprawl patterns
- Healthcare: finding disease subtypes from patient data

### K-Means: Simple and Fast

**The Idea**: Assign points to K clusters by minimizing total distance to cluster centers (centroids).

**Analogy**: Sorting coins into K piles. You start with K random piles. Then:
1. Put each coin in the pile whose center (average position) is closest
2. Recalculate the center of each pile based on new members
3. Repeat until the piles stabilize

**Algorithm**:
```
1. Choose K (number of clusters)
2. Randomly initialize K centroids
3. For each iteration:
   a. Assign each point to the nearest centroid
   b. Recalculate centroid as the mean of all points in the cluster
   c. Stop if centroids don't move (or max iterations reached)
```

**Strengths**:
- Fast: O(nKd) where n = samples, K = clusters, d = dimensions
- Interpretable: each centroid is a real point (or close to one)
- Scales well to large datasets

**Weaknesses**:
- You must specify K upfront (we'll address this shortly)
- Assumes spherical (round) clusters
- Sensitive to outliers (one extreme point can pull a centroid far)
- Can get stuck in local optima (restart multiple times to find the best solution)

### Choosing K: The Elbow Method

How do you know how many clusters to create? Use the **elbow method**:

1. Run K-Means for K = 1, 2, 3, ..., 10
2. Calculate inertia (sum of squared distances from points to nearest centroid) for each K
3. Plot inertia vs K
4. Look for the "elbow" — where inertia stops decreasing sharply

**Example plot**:
```
Inertia
  |
  |     K=1
  |      \
  |       \  K=2
  |        \
  |         \  K=3 <- elbow here
  |          \
  |           \_ K=4, 5, 6, ... (flattens out)
  |_____________________ K
```

At the elbow, adding more clusters stops giving you much benefit. Left of the elbow, adding a cluster significantly reduces inertia. Right of the elbow, adding clusters barely helps.

**In practice**: The elbow is often fuzzy. You might also use domain knowledge ("we have 5 regional markets, so K=5 makes sense").

### Hierarchical Clustering: Building a Family Tree

**The Idea**: Don't decide on K upfront. Instead, build a dendrogram (tree of clusters) showing how data points relate.

**Algorithm** (Agglomerative):
1. Start with each point as its own cluster
2. Repeatedly merge the two closest clusters until you have one big cluster
3. Record the merges in a tree (dendrogram)

**Example**:
```
        |-- Alice
        |       |-- Bob
    ____|       |
   |    |-- Charlie
   |    |       |-- Dana
   |    |__ Elena
```

If you cut the tree at different heights, you get different numbers of clusters. High up: few clusters. Low down: many clusters.

**Linkage Methods** (how to measure "closest" when clusters have multiple points):
- **Single linkage**: distance between closest points in the clusters (can create "chain" effects)
- **Complete linkage**: distance between furthest points (more balanced clusters)
- **Average linkage**: average distance between all pairs (middle ground)
- **Ward linkage**: minimizes variance increase when merging (often the best for round clusters)

**Strengths**:
- No need to specify K upfront
- Interpretable tree structure
- Works with any distance metric

**Weaknesses**:
- Slower: O(n²) or O(n² log n) depending on implementation
- Once clusters are merged, they can't be split (greedy, not globally optimal)
- Hard to visualize beyond 2-3 dimensions

### DBSCAN: Density-Based Clustering

**The Idea**: Clusters are dense regions separated by sparse regions. Grow clusters from core points that have enough neighbors.

**Algorithm**:
1. For each point, count how many points are within distance epsilon (eps)
2. If a point has at least min_samples neighbors, it's a core point
3. Core points with reachable neighbors form a cluster
4. Points not in any cluster are noise/outliers

**Analogy**: A city's neighborhoods. Core points are busy downtown blocks with lots of people. Surrounding blocks are part of the neighbourhood if they're close enough. Isolated farms outside town are noise.

**Parameters**:
- **epsilon (eps)**: maximum distance for two points to be neighbours
- **min_samples**: minimum number of neighbours (including the point itself) to be a core point

**Why DBSCAN is Powerful**:
- Finds clusters of any shape (not just spheres)
- Automatically handles outliers (points that don't fit any cluster)
- No need to specify K
- Works well with spatial data

**Weaknesses**:
- Sensitive to parameter choices (eps and min_samples)
- Struggling with varying cluster densities (hard to set eps when some clusters are tight and others are loose)
- Slower than K-Means: O(n²) in worst case, but O(n log n) with spatial indexing

**How to Set eps and min_samples**:
- Start with min_samples = 2 × dimensions (e.g., for 2D data, min_samples = 4)
- Plot a k-distance graph: for each point, find distance to kth nearest neighbour, sort and plot. The "knee" in the graph is a good starting eps
- Tune from there based on results

### When to Use Which Algorithm

| Algorithm | Best For | You Must Know |
|-----------|----------|---------------|
| **K-Means** | Known K, round clusters, large datasets, speed is critical | Choose K upfront using elbow method or domain knowledge |
| **Hierarchical** | Exploring cluster structure, small datasets, interpretability matters | Budget O(n²) time; choose linkage method carefully |
| **DBSCAN** | Unknown K, irregular shapes, outlier handling, spatial data | Set eps and min_samples carefully; may fail with varying densities |

**Decision flowchart**:
1. Do you know how many clusters you want? → **K-Means**
2. Do you want to visualize cluster hierarchy? → **Hierarchical**
3. Do your clusters have irregular shapes or unknown count? → **DBSCAN**
4. Still unsure? → Try all three and compare silhouette scores

### Quick Check: Clustering

**Question 1**: You have customer data and run K-Means with K=3, K=4, and K=5. Inertia decreases from K=3 (100) to K=4 (92) to K=5 (85). The elbow appears around K=4. Your marketing team wants exactly 5 segments for a campaign. What do you recommend?

*Sample Answer*: Recommend K=4. The elbow suggests diminishing returns beyond K=4. However, gather marketing context: if the 5th segment represents a high-value niche worth special treatment, K=5 might be justified despite worse inertia. Run a silhouette analysis on both—if K=4 has higher silhouette scores, it's a stronger choice. But ultimately, domain knowledge (your marketing team's strategy) should guide the decision.

**Question 2**: You're using DBSCAN to cluster geographic crime data. You notice that downtown has many tight clusters, while suburban areas form looser clusters. Your current eps setting creates many very small clusters downtown but misses suburban clusters. What's the problem and how do you fix it?

*Sample Answer*: The problem is that eps is global—it's the same everywhere. When clusters have different densities, one eps can't work for all regions. Solutions: (1) Use HDBSCAN (a variant that adapts density), (2) Preprocess your data (normalize by local density), or (3) Accept that DBSCAN is limited here and switch to K-Means or Hierarchical clustering with adjusted parameters. This is a known limitation of DBSCAN with heterogeneous densities.

**Question 3**: You apply Hierarchical Clustering to a dataset and get a dendrogram. The tree shows three natural "arms" (groups that branch off early and separately). Is this a strong signal to use K=3 clusters, or should you investigate further?

*Sample Answer*: This is a good starting point but investigate further. The dendrogram shows the structure, but you should also calculate silhouette scores for K=2, K=3, K=4 to see which produces the most cohesive clusters. The "natural" split in the dendrogram is based on linkage criterion (e.g., Ward's variance), not necessarily on how tight and separated the clusters are. A silhouette score tells you if the clustering is actually good, not just visually appealing.

---

## Hackathon Lens 🔨

**Challenge**: Build an interactive K-Means visualizer for beginners.

Users can:
- Upload or use sample data (2D for easy visualization)
- Drag a slider to change K (number of clusters)
- Watch centroids move in real time as K changes
- See inertia plotted vs K (elbow plot builds as they adjust)
- Click "Explain" to see tooltips explaining each step

**Learning objective**: Help beginners develop intuition for how K-Means works and why the elbow method works.

**Bonus**: Add a mode where users manually place K points on a scatter plot and watch K-Means converge toward those points, teaching that K-Means finds local optima.

---

## Key Takeaways

1. **Outlier detection** identifies unusual data points using statistical methods (IQR, z-score) or ML methods (Isolation Forest)
2. **Dimensionality reduction** (PCA) simplifies data while preserving variance, making models faster and more interpretable
3. **Clustering** groups similar points together—choose the algorithm based on whether you know K, whether clusters are regular shapes, and how much you care about interpretability
4. Always **visualize first**: plot your data before deciding on a method
5. Always **compare baselines**: run multiple algorithms and evaluate with silhouette scores or domain-specific metrics
