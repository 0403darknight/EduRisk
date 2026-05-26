import pandas as pd
import numpy as np
import joblib
import json
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (classification_report, accuracy_score,
                              f1_score, precision_score, recall_score,
                              confusion_matrix)
import warnings
warnings.filterwarnings('ignore')

print("Loading dataset...")
df = pd.read_csv('data/dataset.csv')
label_map = {0: 'Dropout', 1: 'Enrolled', 2: 'Graduate'}
df['Target_Label'] = df['Target'].map(label_map)

feature_cols = [c for c in df.columns if c not in ['Target', 'Target_Label']]
X = df[feature_cols]
y = df['Target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)

print("Training models...")
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=7),
    'Decision Tree': DecisionTreeClassifier(max_depth=8, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results[name] = {
        'accuracy': round(accuracy_score(y_test, y_pred) * 100, 2),
        'f1': round(f1_score(y_test, y_pred, average='weighted') * 100, 2),
        'precision': round(precision_score(y_test, y_pred, average='weighted') * 100, 2),
        'recall': round(recall_score(y_test, y_pred, average='weighted') * 100, 2),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
    }
    print(f"  {name}: {results[name]['accuracy']}% accuracy")

best_model_name = max(results, key=lambda k: results[k]['f1'])
best_model = models[best_model_name]
print(f"\nBest model: {best_model_name}")

print("Training K-Means clustering...")
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)
df['Cluster'] = cluster_labels

cluster_profiles = df.groupby('Cluster').agg({
    'Curricular units 1st sem (grade)': 'mean',
    'Curricular units 2nd sem (grade)': 'mean',
    'Attendance (%)': 'mean',
    'Target': lambda x: (x == 0).mean() * 100
}).round(2)
cluster_profiles.columns = ['Avg Sem1 Grade', 'Avg Sem2 Grade', 'Avg Attendance', 'Dropout Rate (%)']

dropout_rates = cluster_profiles['Dropout Rate (%)'].values
risk_map = {}
sorted_idx = np.argsort(dropout_rates)
risk_map[sorted_idx[0]] = 'Low Risk'
risk_map[sorted_idx[1]] = 'Medium Risk'
risk_map[sorted_idx[2]] = 'High Risk'

print("Computing feature importances...")
rf = models['Random Forest']
feat_imp = pd.Series(rf.feature_importances_, index=feature_cols)
feat_imp = feat_imp.sort_values(ascending=False).head(15)

print("Running PCA for cluster visualization...")
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame({
    'PC1': X_pca[:, 0],
    'PC2': X_pca[:, 1],
    'Cluster': cluster_labels,
    'Risk': [risk_map[c] for c in cluster_labels],
    'Target': y.values,
    'Target_Label': df['Target_Label'].values
})

print("Saving all artifacts...")
joblib.dump(best_model, 'models/best_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(kmeans, 'models/kmeans.pkl')
joblib.dump(pca, 'models/pca.pkl')
joblib.dump(feature_cols, 'models/feature_cols.pkl')

results_to_save = {
    'model_results': results,
    'best_model_name': best_model_name,
    'risk_map': {str(k): v for k, v in risk_map.items()},
    'feature_importance': feat_imp.to_dict(),
    'dataset_stats': {
        'total_students': len(df),
        'dropout_count': int((y == 0).sum()),
        'enrolled_count': int((y == 1).sum()),
        'graduate_count': int((y == 2).sum()),
        'dropout_rate': round((y == 0).mean() * 100, 1),
        'graduate_rate': round((y == 2).mean() * 100, 1),
        'avg_sem1_grade': round(df['Curricular units 1st sem (grade)'].mean(), 2),
        'avg_sem2_grade': round(df['Curricular units 2nd sem (grade)'].mean(), 2),
        'avg_attendance': round(df['Attendance (%)'].mean(), 1),
    }
}

with open('models/results.json', 'w') as f:
    json.dump(results_to_save, f, indent=2)

cluster_profiles.to_csv('models/cluster_profiles.csv')
pca_df.to_csv('models/pca_data.csv', index=False)
df.to_csv('data/processed_dataset.csv', index=False)

print("\nAll artifacts saved successfully!")
print(f"Dataset stats: {results_to_save['dataset_stats']}")
