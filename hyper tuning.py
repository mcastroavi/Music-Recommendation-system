##hyper tuning
##0.880 accuracy 

import pandas as pd
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
import numpy as np

# === Load Data ===
data_file = Path('Data_matrix_UserID_TrackID_Score.txt')
ground_truth_file = Path('test2_new.txt')


# Load the datasets
df = pd.read_csv(data_file, sep='|', header=None, names=['userID', 'trackID', 'albumScore', 'artistScore'])
gt = pd.read_csv(ground_truth_file, sep='|', header=None, names=['userID', 'trackID', 'prediction'])

# Convert userID and trackID to strings and remove any leading/trailing whitespace
for col in ['userID', 'trackID']:
    df[col] = df[col].astype(str).str.strip()
    gt[col] = gt[col].astype(str).str.strip()

# Calculate the total score
df['totalScore'] = df['albumScore'] + df['artistScore']

# Merge the data with the ground truth to get the training data
train_df = pd.merge(df, gt, on=['userID', 'trackID'])

# === Normalize Features ===
def zscore(x):
    return (x - x.mean()) / (x.std() + 1e-8)

for col in ['albumScore', 'artistScore', 'totalScore']:
    df[f'{col}_norm'] = df.groupby('userID')[col].transform(zscore)
    train_df[f'{col}_norm'] = train_df.groupby('userID')[col].transform(zscore)

# === Interaction Features ===
df['album_x_artist'] = df['albumScore'] * df['artistScore']
df['album_div_artist'] = df['albumScore'] / (df['artistScore'] + 1e-5)
train_df['album_x_artist'] = train_df['albumScore'] * train_df['artistScore']
train_df['album_div_artist'] = train_df['albumScore'] / (train_df['artistScore'] + 1e-5)

# Rank scores within each user group
df['score_rank'] = df.groupby('userID')['totalScore'].rank(ascending=False)
train_df['score_rank'] = train_df.groupby('userID')['totalScore'].rank(ascending=False)

# === Global Popularity & User Activity ===
track_pop = df.groupby('trackID')['totalScore'].mean().rename('track_popularity')
track_count = df.groupby('trackID').size().rename('track_rating_count')
user_avg = df.groupby('userID')['totalScore'].mean().rename('user_avg_score')

# Merge the popularity and activity features into the main DataFrame
df = df.merge(track_pop, on='trackID', how='left')
df = df.merge(track_count, on='trackID', how='left')
df = df.merge(user_avg, on='userID', how='left')

train_df = train_df.merge(track_pop, on='trackID', how='left')
train_df = train_df.merge(track_count, on='trackID', how='left')
train_df = train_df.merge(user_avg, on='userID', how='left')

# === Final Feature List ===
features = [
    'albumScore', 'artistScore', 'totalScore',
    'albumScore_norm', 'artistScore_norm', 'totalScore_norm',
    'album_x_artist', 'album_div_artist',
    'score_rank', 'track_popularity',
    'track_rating_count', 'user_avg_score'
]

X = train_df[features]
y = train_df['prediction'].astype(int)
# === Hyperparameter Tuning ===
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.03, 0.1],
    'max_depth': [3, 5, 7],
    'subsample': [0.65, 0.75, 0.85]
}

# Initialize the model
model = GradientBoostingClassifier(random_state=0)

# Perform grid search with cross-validation
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
grid_search.fit(X, y)

# Get the best model from grid search
best_model = grid_search.best_estimator_
print(f"Best Parameters: {grid_search.best_params_}")

# === Train-Test Split for Final Evaluation ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
best_model.fit(X_train, y_train)

# === Predict on Test Set ===
y_pred = best_model.predict(X_test)

# === Evaluate on Test Set ===
print("✅ Test Set Accuracy:", accuracy_score(y_test, y_pred))
print("📊 Test Set Classification Report:\n", classification_report(y_test, y_pred))

# === Predict on Entire Dataset ===
df['predicted'] = best_model.predict(df[features])

# === Evaluate on Entire Dataset ===
eval_df = pd.merge(df, gt, on=['userID', 'trackID'], suffixes=('_pred', '_true'))
print("✅ Entire Dataset Accuracy:", accuracy_score(eval_df['prediction'], eval_df['predicted']))
print("📊 Entire Dataset Classification Report:\n", classification_report(eval_df['prediction'], eval_df['predicted']))


## Output detail
# Fitting 5 folds for each of 81 candidates, totalling 405 fits
# Best Parameters: {'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 200, 'subsample': 0.85}
# ✅ Test Set Accuracy: 0.8683333333333333
# 📊 Test Set Classification Report:
#                precision    recall  f1-score   support

#            0       0.84      0.91      0.87       600
#            1       0.90      0.83      0.86       600

#     accuracy                           0.87      1200
#    macro avg       0.87      0.87      0.87      1200
# weighted avg       0.87      0.87      0.87      1200

# ✅ Entire Dataset Accuracy: 0.895
# 📊 Entire Dataset Classification Report:
#                precision    recall  f1-score   support

#            0       0.86      0.94      0.90      3000
#            1       0.93      0.85      0.89      3000

#     accuracy                           0.90      6000
#    macro avg       0.90      0.90      0.89      6000
# weighted avg       0.90      0.90      0.89      6000

# In [5]:


###output csv file  

# === Final Submission ===
df['trackID_combined'] = df['userID'] + '_' + df['trackID']
submission = df[['trackID_combined', 'predicted']]
submission.columns = ['trackID', 'predictor']
submission.to_csv('submission_final_featureboost_v01c.csv', index=False)

# === Compare with Ground Truth for Accuracy ===
# Create a combined column in the ground truth DataFrame
gt['trackID_combined'] = gt['userID'] + '_' + gt['trackID']

# Merge the final submission with the ground truth
ground_truth_comparison = pd.merge(submission, gt, left_on='trackID', right_on='trackID_combined', how='inner')
ground_truth_comparison['correct'] = ground_truth_comparison['predictor'] == ground_truth_comparison['prediction']
final_accuracy = ground_truth_comparison['correct'].mean()

print(f"Final accuracy compared to ground truth: {final_accuracy:.4f}")
## Output detail
Final accuracy compared to ground truth: 0.8950
