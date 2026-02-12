# student_dropout_model.ipynb

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

print("="*60)
print("STUDENT DROPOUT EARLY WARNING SYSTEM")
print("="*60)

# Load data
df = pd.read_csv('xAPI-Edu-Data.csv')
print(f"\n📊 Dataset Shape: {df.shape}")
print(f"📋 Columns: {df.columns.tolist()}")

# Data Cleaning and Preprocessing
print("\n" + "="*60)
print("STEP 1: DATA CLEANING & PREPROCESSING")
print("="*60)

# Check missing values
print(f"\n🔍 Missing Values:\n{df.isnull().sum()}")

# Rename columns for better readability
df.columns = df.columns.str.lower().str.replace(' ', '_')
df.rename(columns={
    'raisedhands': 'raised_hands',
    'visittedresources': 'visited_resources',
    'announcementsview': 'announcements_view',
    'discussion': 'discussion_posts'
}, inplace=True)

# Encode target variable
target_mapping = {'Dropout': 1, 'Graduate': 0, 'Enrolled': 0}
df['dropout_risk'] = df['class'].map(target_mapping)

print(f"\n✅ Target Distribution:")
print(df['dropout_risk'].value_counts())
print(f"Dropout Rate: {df['dropout_risk'].mean():.2%}")

# Feature Engineering
print("\n" + "="*60)
print("STEP 2: FEATURE ENGINEERING")
print("="*60)

# Create engagement score
df['engagement_score'] = (
    df['raised_hands'] * 0.3 + 
    df['visited_resources'] * 0.3 + 
    df['announcements_view'] * 0.2 + 
    df['discussion_posts'] * 0.2
) / 100

# Create low engagement flag
df['low_engagement'] = (df['engagement_score'] < 0.3).astype(int)

# Create parent involvement score
df['parent_involvement'] = df['parentanswerssurvery'].map({
    'Yes': 2, 'No': 0
}).fillna(1)  # Under Some is 1

# Create semester progress indicator
df['early_semester_proxy'] = (
    (df['raised_hands'] > 50) & 
    (df['visited_resources'] > 50) & 
    (df['announcements_view'] > 20)
).astype(int)

print("✅ Created Features:")
print("  - engagement_score")
print("  - low_engagement")
print("  - parent_involvement")
print("  - early_semester_proxy")

# Feature Selection
print("\n" + "="*60)
print("STEP 3: FEATURE SELECTION")
print("="*60)

categorical_features = ['gender', 'nationality', 'placeofbirth', 'stageid', 
                        'gradeid', 'sectionid', 'topic', 'semester', 
                        'relation', 'parentanswerssurvery', 'parentschoolsatisfaction', 
                        'studentabsencedays']

numerical_features = ['raised_hands', 'visited_resources', 'announcements_view', 
                     'discussion_posts', 'engagement_score', 'low_engagement', 
                     'parent_involvement', 'early_semester_proxy']

# Keep only early-semester features (simulate early detection)
early_features = ['gender', 'nationality', 'stageid', 'gradeid', 'sectionid',
                 'raised_hands', 'visited_resources', 'announcements_view', 
                 'discussion_posts', 'relation', 'parentanswerssurvery',
                 'parentschoolsatisfaction', 'engagement_score', 
                 'low_engagement', 'parent_involvement', 'early_semester_proxy']

X = df[early_features]
y = df['dropout_risk']

print(f"✅ Using {len(early_features)} early-semester features")

# Preprocessing Pipeline
print("\n" + "="*60)
print("STEP 4: BUILDING PREPROCESSING PIPELINE")
print("="*60)

# Identify categorical and numerical columns in early_features
cat_cols = X.select_dtypes(include=['object']).columns.tolist()
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"📊 Categorical features: {len(cat_cols)}")
print(f"📊 Numerical features: {len(num_cols)}")

# Create preprocessing pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# Numerical pipeline
numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical pipeline
categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(drop='first', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', numerical_pipeline, num_cols),
    ('cat', categorical_pipeline, cat_cols)
])

print("✅ Preprocessing pipeline created")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n📊 Training set: {X_train.shape}")
print(f"📊 Test set: {X_test.shape}")

# Model Training
print("\n" + "="*60)
print("STEP 5: MODEL TRAINING & OPTIMIZATION")
print("="*60)

# Create pipeline with Random Forest
rf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        class_weight='balanced'
    ))
])

# Train model
print("\n🌳 Training Random Forest model...")
rf_pipeline.fit(X_train, y_train)

# Make predictions
y_pred = rf_pipeline.predict(X_test)
y_pred_proba = rf_pipeline.predict_proba(X_test)[:, 1]

print("✅ Model training complete!")

# Model Evaluation
print("\n" + "="*60)
print("STEP 6: MODEL EVALUATION")
print("="*60)

# Classification Report
print("\n📈 Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Continue', 'Dropout']))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\n📊 Confusion Matrix:")
print(f"True Negatives: {cm[0,0]} | False Positives: {cm[0,1]}")
print(f"False Negatives: {cm[1,0]} | True Positives: {cm[1,1]}")

# ROC-AUC Score
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"\n🎯 ROC-AUC Score: {roc_auc:.3f}")

# Cross-validation score
cv_scores = cross_val_score(rf_pipeline, X_train, y_train, cv=5, scoring='roc_auc')
print(f"📊 5-Fold CV ROC-AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")

# Feature Importance
print("\n" + "="*60)
print("STEP 7: FEATURE IMPORTANCE ANALYSIS")
print("="*60)

# Get feature names after preprocessing
feature_names = (
    num_cols + 
    rf_pipeline.named_steps['preprocessor']
    .named_transformers_['cat']
    .named_steps['onehot']
    .get_feature_names_out(cat_cols).tolist()
)

importances = rf_pipeline.named_steps['classifier'].feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': feature_names[:len(importances)],
    'importance': importances
}).sort_values('importance', ascending=False)

print("\n🔑 Top 10 Most Important Features:")
print(feature_importance_df.head(10).to_string(index=False))

# Visualize feature importance
plt.figure(figsize=(12, 6))
top_features = feature_importance_df.head(10)
plt.barh(range(len(top_features)), top_features['importance'].values)
plt.yticks(range(len(top_features)), top_features['feature'].values)
plt.xlabel('Importance')
plt.title('Top 10 Features for Dropout Prediction')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

# Risk Thresholds
print("\n" + "="*60)
print("STEP 8: SETTING RISK THRESHOLDS")
print("="*60)

# Define risk levels based on probability thresholds
risk_levels = {
    'Low': 0.3,
    'Medium': 0.6,
    'High': 1.0
}

print(f"🎯 Risk Thresholds:")
print(f"   Low Risk: 0 - {risk_levels['Low']:.0%}")
print(f"   Medium Risk: {risk_levels['Low']:.0%} - {risk_levels['Medium']:.0%}")
print(f"   High Risk: {risk_levels['Medium']:.0%} - {risk_levels['High']:.0%}")

# Save model and preprocessor
print("\n" + "="*60)
print("STEP 9: SAVING MODEL & PIPELINE")
print("="*60)

joblib.dump(rf_pipeline, 'student_dropout_model.joblib')
print("✅ Model saved as 'student_dropout_model.joblib'")

# Save preprocessing pipeline separately
joblib.dump(preprocessor, 'preprocessing_pipeline.joblib')
print("✅ Preprocessing pipeline saved as 'preprocessing_pipeline.joblib'")

# Generate predictions for all students
all_predictions = rf_pipeline.predict_proba(X)[:, 1]

# Create risk labels
risk_labels = []
for prob in all_predictions:
    if prob < risk_levels['Low']:
        risk_labels.append('Low')
    elif prob < risk_levels['Medium']:
        risk_labels.append('Medium')
    else:
        risk_labels.append('High')

# Create predictions dataframe
predictions_df = pd.DataFrame({
    'student_id': df.index + 1000,  # Create student IDs
    'risk_score': all_predictions,
    'risk_label': risk_labels,
    'predicted_dropout': (all_predictions >= 0.5).astype(int),
    'actual_dropout': df['dropout_risk'].values
})

print(f"\n📊 Prediction Distribution:")
print(predictions_df['risk_label'].value_counts())
print(f"\n🎯 Students flagged as high-risk: {sum(predictions_df['risk_label'] == 'High')}")
print(f"   Dropout students caught: {sum((predictions_df['risk_label'] == 'High') & (predictions_df['actual_dropout'] == 1))}")

# Save predictions
predictions_df.to_csv('student_predictions.csv', index=False)
print("\n✅ Predictions saved as 'student_predictions.csv'")

print("\n" + "="*60)
print("🎉 MODEL TRAINING COMPLETE! 🎉")
print("="*60)