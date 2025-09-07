# %% [markdown]
"""
# Data Analysis with Percent Format
This notebook demonstrates using Python files as Jupyter notebooks
using the percent format. You can run this in VS Code or convert it
to `.ipynb` format.
"""

# %% [markdown]
"""
## 1. Setup and Imports
Configure the environment and load required libraries
"""

# %% {"tags": ["setup"]}
# Enable auto-reload of modules
from IPython import get_ipython

if get_ipython() is not None:
    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")

# %% {"tags": ["imports"]}
import sys
import warnings
from pathlib import Path
import os

# Add parent directory to path for imports (Jupyter-compatible)
current_dir = Path.cwd()
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

# Alternative approach using os.getcwd()
# sys.path.insert(0, str(Path(os.getcwd()).parent))

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import data science libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configure visualization settings
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")

# Display options
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 50)

print("✓ Environment configured")
print(f"✓ Current working directory: {current_dir}")
print(f"✓ Parent directory added to path: {parent_dir}")

# %% [markdown]
"""
## 2. Load Data
Import project utilities and load sample data
"""

# %%
# Import project modules
from src.data_utils import load_sample_data, summarize_data, clean_data

# Generate sample data
df = load_sample_data(n_rows=1000)

print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print("\nFirst 5 rows:")
df.head()

# %% [markdown]
"""
## 3. Data Exploration
Examine the structure and quality of the data
"""

# %% {"tags": ["exploration"]}
# Get summary statistics
summary = summarize_data(df)

print("=" * 50)
print("DATA SUMMARY")
print("=" * 50)

for key, value in summary.items():
    if key == "numeric_stats":
        continue
    elif key == "dtypes":
        print(f"\n{key}:")
        for col, dtype in value.items():
            print(f"  {col}: {dtype}")
    elif key == "missing":
        print(f"\n{key}:")
        for col, count in value.items():
            if count > 0:
                print(f"  {col}: {count}")
    else:
        print(f"{key}: {value}")

# %%
# Check data types and memory usage
print("\nMemory Usage by Column:")
print("-" * 30)
memory_usage = df.memory_usage(deep=True)
for col, mem in memory_usage.items():
    print(f"{col:15} {mem / 1024:.2f} KB")
print(f"{'Total:':15} {memory_usage.sum() / 1024:.2f} KB")

# %% [markdown]
"""
## 4. Data Visualization
Create visualizations to understand patterns in the data
"""

# %% {"tags": ["visualization"]}
# Create comprehensive visualization
fig = plt.figure(figsize=(15, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. Value distribution
ax1 = fig.add_subplot(gs[0, :2])
df["value"].hist(bins=50, ax=ax1, edgecolor="black", alpha=0.7)
ax1.axvline(
    df["value"].mean(),
    color="red",
    linestyle="--",
    label=f"Mean: {df['value'].mean():.2f}",
)
ax1.axvline(
    df["value"].median(),
    color="green",
    linestyle="--",
    label=f"Median: {df['value'].median():.2f}",
)
ax1.set_title("Value Distribution", fontsize=14, fontweight="bold")
ax1.set_xlabel("Value")
ax1.set_ylabel("Frequency")
ax1.legend()
ax1.grid(alpha=0.3)

# 2. Category distribution
ax2 = fig.add_subplot(gs[0, 2])
category_counts = df["category"].value_counts()
colors = sns.color_palette("husl", len(category_counts))
ax2.pie(
    category_counts.values,
    labels=category_counts.index,
    autopct="%1.1f%%",
    colors=colors,
)
ax2.set_title("Category Distribution", fontsize=14, fontweight="bold")

# 3. Time series
ax3 = fig.add_subplot(gs[1, :])
df_time = df.set_index("timestamp")["value"].resample("D").mean()
ax3.plot(df_time.index, df_time.values, linewidth=2, color="navy")
ax3.fill_between(df_time.index, df_time.values, alpha=0.3)
ax3.set_title("Average Daily Values", fontsize=14, fontweight="bold")
ax3.set_xlabel("Date")
ax3.set_ylabel("Average Value")
ax3.grid(alpha=0.3)

# 4. Box plot by category
ax4 = fig.add_subplot(gs[2, 0])
df.boxplot(column="value", by="category", ax=ax4)
ax4.set_title("Value Distribution by Category", fontsize=14, fontweight="bold")
ax4.set_xlabel("Category")
ax4.set_ylabel("Value")
plt.sca(ax4)
plt.xticks(rotation=0)

# 5. Validity rate by category
ax5 = fig.add_subplot(gs[2, 1])
validity_by_cat = df.groupby("category")["is_valid"].mean() * 100
validity_by_cat.plot(kind="bar", ax=ax5, color=colors)
ax5.set_title("Validity Rate by Category", fontsize=14, fontweight="bold")
ax5.set_xlabel("Category")
ax5.set_ylabel("Validity Rate (%)")
ax5.set_ylim([0, 100])
plt.sca(ax5)
plt.xticks(rotation=0)

# 6. Correlation heatmap
ax6 = fig.add_subplot(gs[2, 2])
# Create numeric encoding for category
df["category_encoded"] = df["category"].astype("category").cat.codes
corr_data = df[["value", "category_encoded", "is_valid"]].corr()
sns.heatmap(corr_data, annot=True, cmap="coolwarm", center=0, ax=ax6, vmin=-1, vmax=1)
ax6.set_title("Correlation Matrix", fontsize=14, fontweight="bold")

plt.suptitle("Data Analysis Dashboard", fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()

# %% [markdown]
"""
## 5. Statistical Analysis
Perform statistical tests and deeper analysis
"""

# %%
# Statistical tests by category
from scipy import stats

print("Statistical Analysis by Category")
print("=" * 50)

categories = df["category"].unique()
for i, cat1 in enumerate(categories):
    for cat2 in categories[i + 1 :]:
        data1 = df[df["category"] == cat1]["value"]
        data2 = df[df["category"] == cat2]["value"]

        # T-test
        t_stat, p_value = stats.ttest_ind(data1, data2)

        print(f"\n{cat1} vs {cat2}:")
        print(f"  Mean difference: {data1.mean() - data2.mean():.2f}")
        print(f"  T-statistic: {t_stat:.4f}")
        print(f"  P-value: {p_value:.4f}")
        print(f"  Significant: {'Yes' if p_value < 0.05 else 'No'} (α=0.05)")

# %%
# Check for outliers using IQR method
Q1 = df["value"].quantile(0.25)
Q3 = df["value"].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df["value"] < lower_bound) | (df["value"] > upper_bound)]

print(f"\nOutlier Detection (IQR Method)")
print(f"=" * 40)
print(f"Lower bound: {lower_bound:.2f}")
print(f"Upper bound: {upper_bound:.2f}")
print(f"Number of outliers: {len(outliers)} ({len(outliers) / len(df) * 100:.1f}%)")
print(f"\nOutlier distribution by category:")
print(outliers["category"].value_counts())

# %% [markdown]
"""
## 6. Data Cleaning and Preprocessing
Clean the data and prepare for modeling
"""

# %%
# Clean the data
df_cleaned = clean_data(df, drop_nulls=False, drop_duplicates=True)

print("Data Cleaning Results")
print("=" * 40)
print(f"Original shape: {df.shape}")
print(f"Cleaned shape: {df_cleaned.shape}")
print(f"Rows removed: {len(df) - len(df_cleaned)}")

# Add derived features
df_cleaned["value_normalized"] = (
    df_cleaned["value"] - df_cleaned["value"].mean()
) / df_cleaned["value"].std()
df_cleaned["hour"] = df_cleaned["timestamp"].dt.hour
df_cleaned["day_of_week"] = df_cleaned["timestamp"].dt.dayofweek
df_cleaned["is_weekend"] = df_cleaned["day_of_week"].isin([5, 6]).astype(int)

print(f"\nNew features added:")
print(f"  - value_normalized (z-score)")
print(f"  - hour (0-23)")
print(f"  - day_of_week (0=Monday, 6=Sunday)")
print(f"  - is_weekend (0/1)")

# %% [markdown]
"""
## 7. Export Results
Save cleaned data and visualizations
"""

# %%
# Save cleaned data
output_dir = Path("../outputs")
output_dir.mkdir(exist_ok=True)

# Export to CSV
csv_file = output_dir / "cleaned_data.csv"
df_cleaned.to_csv(csv_file, index=False)
print(f"✓ Cleaned data saved to: {csv_file}")

# Export summary statistics
summary_file = output_dir / "data_summary.json"
import json

with open(summary_file, "w") as f:
    # Convert numpy types to native Python types for JSON serialization
    summary_json = {}
    for key, value in summary.items():
        if key == "numeric_stats":
            continue
        elif isinstance(value, dict):
            summary_json[key] = {k: str(v) for k, v in value.items()}
        else:
            summary_json[key] = str(value)
    json.dump(summary_json, f, indent=2)
print(f"✓ Summary statistics saved to: {summary_file}")

# %% [markdown]
"""
## 8. Key Findings and Conclusions

### Summary of Analysis:
1. **Data Quality**: The dataset has minimal missing values and is generally clean
2. **Distribution**: Values follow approximately normal distribution with mean ~500
3. **Categories**: Fairly balanced distribution across categories A, B, C, D
4. **Temporal Patterns**: Some daily variation in average values
5. **Outliers**: Approximately 5% of data points are statistical outliers
6. **Validity**: High validity rate (~90%) across all categories

### Next Steps:
- Investigate the cause of outliers
- Build predictive models using the cleaned dataset
- Perform time series analysis on temporal patterns
- Create automated reporting pipeline

### Notes:
This notebook can be:
- Run directly in VS Code (no browser needed)
- Converted to `.ipynb` format using jupytext
- Executed from command line as a Python script
- Version controlled with clean diffs
"""

# %%
print("\n" + "=" * 50)
print("ANALYSIS COMPLETE")
print("=" * 50)
print(f"Total runtime: Check notebook metadata")
print(f"Environment: {sys.version}")
print(f"Pandas version: {pd.__version__}")
print(f"NumPy version: {np.__version__}")

# %%
