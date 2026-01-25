# =============================================================================
# COMPLETE EDA FOR CLEVELAND HEART DISEASE DATASET (UCI)
# Includes: distributions by target, stats, missing values, correlations,
#           categorical deep dive, key scatter plots, and insights
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# =============================================================================
# 1. LOAD AND CLEAN THE DATA
# =============================================================================

url = "Heart_disease_cleveland_new.csv"

columns = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
]

df = pd.read_csv(url, names=columns)

print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")

# =============================================================================
# 2. MISSING VALUES
# =============================================================================

print("\nMissing values:")
print(df.isnull().sum())

plt.figure(figsize=(10, 4))
sns.heatmap(df.isnull(), cbar=True, yticklabels=False, cmap='viridis')
plt.title('Missing Values Heatmap')
plt.show()

# Simple imputation for visualization (median for ca, mode for thal)
df_viz = df.copy()
df_viz['ca'].fillna(df_viz['ca'].median(), inplace=True)
df_viz['thal'].fillna(df_viz['thal'].mode()[0], inplace=True)

# =============================================================================
# 3. TARGET DISTRIBUTION
# =============================================================================

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
df['target'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['#66b3ff','#ff9999'])
plt.title('Target Distribution (0 = No Disease, 1 = Disease)')
plt.ylabel('')

plt.subplot(1, 2, 2)
sns.countplot(data=df, x='target', palette=['#66b3ff','#ff9999'])
plt.title('Count of Heart Disease Cases')
plt.xlabel('Target')
plt.xticks([0, 1], ['No Disease', 'Disease'])
for i, v in enumerate(df['target'].value_counts()):
    plt.text(i, v + 2, str(v), ha='center', fontweight='bold')
plt.show()

print(f"Disease prevalence: {df['target'].mean():.1%}")

# =============================================================================
# 4. SUMMARY STATISTICS BY TARGET
# =============================================================================

print("\nSummary statistics by target:")

# =============================================================================
# 5. UNIVARIATE DISTRIBUTIONS + BOXPLOTS BY TARGET
# =============================================================================

continuous_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

fig, axes = plt.subplots(len(continuous_features), 2, figsize=(14, 4*len(continuous_features)))
for i, col in enumerate(continuous_features):
    # Histogram + KDE
    sns.histplot(data=df_viz, x=col, hue='target', kde=True, ax=axes[i,0], palette=['#66b3ff','#ff9999'], alpha=0.7)
    axes[i,0].set_title(f'Distribution of {col}')
    
    # Boxplot
    sns.boxplot(data=df_viz, x='target', y=col, ax=axes[i,1], palette=['#66b3ff','#ff9999'])
    axes[i,1].set_title(f'{col} by Target')
    axes[i,1].set_xlabel('Target')

plt.tight_layout()
plt.show()

# =============================================================================
# 6. CATEGORICAL FEATURES - DISEASE PREVALENCE
# =============================================================================

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.ravel()

for i, col in enumerate(categorical_features):
    # Cross tab
    ct = pd.crosstab(df_viz[col], df_viz['target'], normalize='index')
    ct.plot(kind='bar', stacked=True, ax=axes[i], color=['#66b3ff','#ff9999'], alpha=0.8)
    axes[i].set_title(f'Disease % by {col}')
    axes[i].legend(title='Disease', labels=['No', 'Yes'])
    axes[i].set_ylabel('Proportion')

plt.suptitle('Disease Prevalence by Categorical Features', fontsize=16, y=1.02)
plt.tight_layout()
plt.show()

# Feature name mapping for better labels
name_map = {
    'sex': {0: 'Female', 1: 'Male'},
    'cp': {1: 'Typical Angina', 2: 'Atypical Angina', 3: 'Non-anginal Pain', 4: 'Asymptomatic'},
    'fbs': {0: '≤120 mg/dl', 1: '>120 mg/dl'},
    'restecg': {0: 'Normal', 1: 'ST-T abnormality', 2: 'LV hypertrophy'},
    'exang': {0: 'No', 1: 'Yes'},
    'slope': {1: 'Upsloping', 2: 'Flat', 3: 'Downsloping'},
    'ca': {0: '0 vessels', 1: '1 vessel', 2: '2 vessels', 3: '3 vessels'},
    'thal': {3: 'Normal', 6: 'Fixed defect', 7: 'Reversible defect'}
}

# =============================================================================
# 7. CORRELATION MATRIX
# =============================================================================

plt.figure(figsize=(12, 8))
corr = df_viz.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

print("\nTop 5 features most correlated with target:")
print(corr['target'].abs().sort_values(ascending=False).head(6))

# =============================================================================
# 8. KEY BIVARIATE SCATTER PLOTS
# =============================================================================

plt.figure(figsize=(18, 5))

plt.subplot(1, 3, 1)
sns.scatterplot(data=df_viz, x='age', y='thalach', hue='target', palette=['#66b3ff','#ff9999'], alpha=0.8)
plt.title('Age vs Max Heart Rate (thalach)')

plt.subplot(1, 3, 2)
sns.scatterplot(data=df_viz, x='age', y='oldpeak', hue='target', palette=['#66b3ff','#ff9999'], alpha=0.8)
plt.title('Age vs ST Depression (oldpeak)')

plt.subplot(1, 3, 3)
sns.scatterplot(data=df_viz, x='thalach', y='oldpeak', hue='target', palette=['#66b3ff','#ff9999'], alpha=0.8)
plt.title('Max HR vs ST Depression')

plt.tight_layout()
plt.show()

# =============================================================================
# 9. AGE BINS & DISEASE RISK
# =============================================================================

df_viz['age_group'] = pd.cut(df_viz['age'], bins=[29, 40, 50, 60, 70, 80], labels=['<40', '40-49', '50-59', '60-69', '70+'])

plt.figure(figsize=(8, 5))
age_risk = df_viz.groupby('age_group')['target'].mean()
age_risk.plot(kind='bar', color='salmon')
plt.title('Heart Disease Risk by Age Group')
plt.ylabel('Disease Prevalence')
plt.xticks(rotation=0)
for i, v in enumerate(age_risk):
    plt.text(i, v + 0.01, f'{v:.1%}', ha='center', fontweight='bold')
plt.show()

# =============================================================================
# FINAL INSIGHTS
# =============================================================================

print("\n" + "="*60)
print("KEY INSIGHTS FROM EDA")
print("="*60)
print("• Strongest predictors: cp, thal, ca, oldpeak, exang, thalach, slope")
print("• Men have much higher disease rate than women")
print("• Asymptomatic chest pain (cp=4) → very high risk")
print("• Reversible defect in thal → very high risk")
print("• Higher max heart rate (thalach) → lower risk (especially in younger patients)")
print("• Number of major vessels (ca) = 0 → low risk, ≥1 → high risk")
print("• Disease risk increases steadily with age")
print("• Only 4 missing values in ca, 2 in thal → safe to impute")
print("="*60)