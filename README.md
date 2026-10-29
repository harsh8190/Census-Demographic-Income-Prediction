# Adult Census Income Classification and Demographic Clustering

A machine learning project that predicts whether a person's annual income is **greater than $50K** using the Adult Census Income dataset. It also explores demographic patterns with K-Means clustering.

## Objectives

- Predict income class (`<=50K` or `>50K`).
- Compare five supervised machine learning models.
- Evaluate models using different train/test splits.
- Visualize model performance and explore demographic clusters.
- Generate example predictions and probabilities for the `>50K` class.

## Algorithms Used

| Model | Configuration |
|---|---|
| Logistic Regression | Standardized features |
| Decision Tree | Maximum depth = 10 |
| K-Nearest Neighbors (KNN) | 15 neighbors; standardized features |
| Linear Support Vector Classifier (LinearSVC) | Standardized features |
| Random Forest | 100 trees; maximum depth = 12 |

Models are evaluated using accuracy, precision, recall, and F1-score. The project compares stratified 80:20, 70:30, and 75:25 train/test splits.

## Dataset

The project expects a CSV file named `adult.csv`. Place it in the project directory or update the dataset path in the Python script.

Preprocessing includes:
- Replacing `?` with missing values.
- Removing the `education` column from model features.
- Encoding income as `0` for `<=50K` and `1` for `>50K`.
- One-hot encoding categorical features.
- Filling missing numeric values with column medians.

## Requirements

- Python 3.9 or later recommended
- pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn

Install the required packages:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## How to Run

1. Place `adult.csv` and the project Python script in your working directory.
2. Update the dataset path if necessary.
3. Install the required dependencies.
4. Run the script:

```bash
python your_script_name.py
```

Replace `your_script_name.py` with your actual Python filename.

## Project Workflow

1. Load the Adult Census Income dataset.
2. Clean missing-value markers and prepare the income target.
3. Encode categorical features and handle missing values.
4. Create stratified train/test splits.
5. Train five classification models.
6. Calculate accuracy, precision, recall, and F1-score.
7. Generate model-performance charts and a confusion matrix.
8. Apply K-Means clustering with two clusters.
9. Use PCA to visualize the clusters in two dimensions.
10. Generate example Random Forest predictions and income probabilities.

## Outputs

Depending on the filenames configured in the Python script, the project generates:

- CSV files containing evaluation results for individual splits and combined results.
- An income distribution plot.
- An accuracy comparison across train/test splits.
- A precision, recall, and F1-score comparison for the 80:20 split.
- A confusion matrix for the model with the best F1-score.
- A PCA scatter plot of the K-Means clusters.
- Cluster labels and high-income-rate comparisons.
- Example demographic predictions and predicted probabilities.

Check the output paths in the Python script for the exact filenames and locations.

## Evaluation Metrics

- **Accuracy:** Proportion of predictions that are correct.
- **Precision:** Proportion of predicted `>50K` records that actually belong to that class.
- **Recall:** Proportion of actual `>50K` records correctly identified.
- **F1-score:** Harmonic mean of precision and recall.

These metrics should be reviewed together, especially if the income classes are imbalanced.

## Limitations and Responsible Use

- **Actual results:** Model scores depend on the dataset and configuration. Use the generated evaluation files rather than treating hard-coded example values as verified results.
- **Preprocessing leakage:** If preprocessing statistics are calculated before the train/test split, information from the test set can influence the model. A scikit-learn `Pipeline` and `ColumnTransformer` can help prevent this.
- **Clustering:** Cluster labels do not inherently represent low- or high-income groups. Review cluster summaries before interpreting them.
- **PCA:** A two-dimensional projection cannot preserve every relationship in the original feature space.
- **Fairness:** Income predictions may reflect historical inequalities. Do not use this project as the sole basis for employment, lending, or other consequential decisions.
- **Reproducibility:** Record the dataset version, package versions, random seeds, and actual evaluation metrics.

## Future Improvements

- Use a leakage-safe preprocessing pipeline.
- Tune model hyperparameters with cross-validation.
- Add ROC-AUC and precision-recall curves.
- Include feature-importance or model-explanation analysis.
- Evaluate alternative clustering configurations.
- Add automated tests and configurable input/output paths.
