"""Create Jupyter notebooks programmatically."""

from pathlib import Path
from typing import List, Optional, Union

import nbformat as nbf


class NotebookBuilder:
    """Builder class for creating Jupyter notebooks programmatically."""

    def __init__(self, title: str = "Untitled"):
        """Initialize notebook builder.

        Args:
            title: Notebook title
        """
        self.notebook = nbf.v4.new_notebook()
        self.title = title

        # Add title as first cell
        if title != "Untitled":
            self.add_markdown_cell(f"# {title}")

    def add_markdown_cell(self, content: str) -> "NotebookBuilder":
        """Add markdown cell to notebook.

        Args:
            content: Markdown content

        Returns:
            Self for chaining
        """
        self.notebook.cells.append(nbf.v4.new_markdown_cell(content))
        return self

    def add_code_cell(
        self, code: str, execution_count: Optional[int] = None
    ) -> "NotebookBuilder":
        """Add code cell to notebook.

        Args:
            code: Python code
            execution_count: Cell execution count

        Returns:
            Self for chaining
        """
        cell = nbf.v4.new_code_cell(code)
        if execution_count is not None:
            cell.execution_count = execution_count
        self.notebook.cells.append(cell)
        return self

    def add_import_cell(self, imports: List[str]) -> "NotebookBuilder":
        """Add standard imports cell.

        Args:
            imports: List of import statements

        Returns:
            Self for chaining
        """
        import_code = "\n".join(imports)
        self.add_code_cell(import_code)
        return self

    def add_setup_cell(self) -> "NotebookBuilder":
        """Add standard setup cell with common configurations.

        Returns:
            Self for chaining
        """
        setup_code = """# Notebook setup
import warnings
warnings.filterwarnings('ignore')

# Auto-reload modules
%load_ext autoreload
%autoreload 2

# Display settings
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-darkgrid')
%matplotlib inline"""

        self.add_code_cell(setup_code)
        return self

    def save(self, filepath: Union[str, Path]) -> Path:
        """Save notebook to file.

        Args:
            filepath: Output file path

        Returns:
            Path to saved notebook
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w") as f:
            nbf.write(self.notebook, f)

        return filepath

    @classmethod
    def create_template(cls, template_type: str = "analysis") -> "NotebookBuilder":
        """Create notebook from template.

        Args:
            template_type: Type of template ('analysis', 'visualization', 'model')

        Returns:
            NotebookBuilder with template content
        """
        templates = {
            "analysis": cls._analysis_template,
            "visualization": cls._visualization_template,
            "model": cls._model_template,
        }

        if template_type not in templates:
            raise ValueError(f"Unknown template: {template_type}")

        return templates[template_type]()

    @classmethod
    def _analysis_template(cls) -> "NotebookBuilder":
        """Create data analysis template."""
        nb = cls("Data Analysis Template")

        nb.add_markdown_cell("## 1. Setup")
        nb.add_setup_cell()

        nb.add_markdown_cell("## 2. Import Libraries")
        nb.add_import_cell(
            [
                "import numpy as np",
                "import pandas as pd",
                "import matplotlib.pyplot as plt",
                "import seaborn as sns",
                "from pathlib import Path",
                "",
                "# Import project modules",
                "import sys",
                "sys.path.append('../')",
                "from src.data_utils import load_sample_data, summarize_data",
            ]
        )

        nb.add_markdown_cell("## 3. Load Data")
        nb.add_code_cell("""# Load sample data
df = load_sample_data(n_rows=1000)
print(f"Data shape: {df.shape}")
df.head()""")

        nb.add_markdown_cell("## 4. Exploratory Data Analysis")
        nb.add_code_cell("""# Get summary statistics
summary = summarize_data(df)
for key, value in summary.items():
    if key != 'numeric_stats':
        print(f"{key}: {value}")""")

        nb.add_markdown_cell("## 5. Data Visualization")
        nb.add_code_cell("""# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Distribution plot
df['value'].hist(ax=axes[0, 0], bins=30)
axes[0, 0].set_title('Value Distribution')

# Category counts
df['category'].value_counts().plot(kind='bar', ax=axes[0, 1])
axes[0, 1].set_title('Category Counts')

# Time series
df.set_index('timestamp')['value'].plot(ax=axes[1, 0])
axes[1, 0].set_title('Values Over Time')

# Box plot by category
df.boxplot(column='value', by='category', ax=axes[1, 1])
axes[1, 1].set_title('Value by Category')

plt.tight_layout()
plt.show()""")

        nb.add_markdown_cell("## 6. Conclusions")
        nb.add_markdown_cell("- Key finding 1\n- Key finding 2\n- Key finding 3")

        return nb

    @classmethod
    def _visualization_template(cls) -> "NotebookBuilder":
        """Create visualization template."""
        nb = cls("Visualization Template")

        nb.add_markdown_cell("## Setup")
        nb.add_import_cell(
            [
                "import numpy as np",
                "import pandas as pd",
                "import matplotlib.pyplot as plt",
                "import seaborn as sns",
                "import plotly.express as px",
                "import plotly.graph_objects as go",
                "",
                "# Configure plotting",
                "plt.rcParams['figure.figsize'] = (10, 6)",
                "plt.rcParams['font.size'] = 12",
                "sns.set_palette('husl')",
            ]
        )

        nb.add_markdown_cell("## Generate Sample Data")
        nb.add_code_cell("""# Create sample dataset
np.random.seed(42)
n = 500

data = pd.DataFrame({
    'x': np.random.randn(n),
    'y': np.random.randn(n),
    'category': np.random.choice(['A', 'B', 'C'], n),
    'size': np.random.uniform(10, 100, n)
})

data.head()""")

        nb.add_markdown_cell("## Static Visualizations (Matplotlib/Seaborn)")
        nb.add_code_cell("""# Create multiple plot types
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Scatter plot
axes[0, 0].scatter(data['x'], data['y'], alpha=0.5)
axes[0, 0].set_title('Scatter Plot')

# Histogram
axes[0, 1].hist(data['x'], bins=30, edgecolor='black')
axes[0, 1].set_title('Histogram')

# Box plot
data.boxplot(column='x', by='category', ax=axes[0, 2])
axes[0, 2].set_title('Box Plot')

# Violin plot
sns.violinplot(data=data, x='category', y='y', ax=axes[1, 0])
axes[1, 0].set_title('Violin Plot')

# Heatmap
corr_matrix = data[['x', 'y', 'size']].corr()
sns.heatmap(corr_matrix, annot=True, ax=axes[1, 1])
axes[1, 1].set_title('Correlation Heatmap')

# KDE plot
sns.kdeplot(data=data, x='x', y='y', ax=axes[1, 2])
axes[1, 2].set_title('KDE Plot')

plt.tight_layout()
plt.show()""")

        nb.add_markdown_cell("## Interactive Visualizations (Plotly)")
        nb.add_code_cell("""# Interactive scatter plot
fig = px.scatter(
    data, x='x', y='y',
    color='category', size='size',
    title='Interactive Scatter Plot',
    hover_data=['size']
)
fig.show()""")

        return nb

    @classmethod
    def _model_template(cls) -> "NotebookBuilder":
        """Create machine learning model template."""
        nb = cls("ML Model Template")

        nb.add_markdown_cell("## 1. Setup")
        nb.add_import_cell(
            [
                "import numpy as np",
                "import pandas as pd",
                "import matplotlib.pyplot as plt",
                "from sklearn.model_selection import train_test_split",
                "from sklearn.preprocessing import StandardScaler",
                "from sklearn.metrics import classification_report, confusion_matrix",
                "from sklearn.ensemble import RandomForestClassifier",
            ]
        )

        nb.add_markdown_cell("## 2. Data Preparation")
        nb.add_code_cell("""# Generate synthetic data
from sklearn.datasets import make_classification

X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    n_classes=2,
    random_state=42
)

# Convert to DataFrame
feature_names = [f'feature_{i}' for i in range(X.shape[1])]
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

print(f"Dataset shape: {df.shape}")
print(f"Class distribution:\\n{df['target'].value_counts()}")""")

        nb.add_markdown_cell("## 3. Train/Test Split")
        nb.add_code_cell("""# Split data
X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")""")

        nb.add_markdown_cell("## 4. Feature Scaling")
        nb.add_code_cell("""# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)""")

        nb.add_markdown_cell("## 5. Model Training")
        nb.add_code_cell("""# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)""")

        nb.add_markdown_cell("## 6. Model Evaluation")
        nb.add_code_cell("""# Evaluation metrics
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()""")

        nb.add_markdown_cell("## 7. Feature Importance")
        nb.add_code_cell("""# Plot feature importance
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False).head(10)

plt.figure(figsize=(10, 6))
plt.barh(importance_df['feature'], importance_df['importance'])
plt.xlabel('Importance')
plt.title('Top 10 Feature Importances')
plt.gca().invert_yaxis()
plt.show()""")

        return nb


def create_sample_notebooks(output_dir: Union[str, Path] = "notebooks"):
    """Create sample notebooks demonstrating different use cases.

    Args:
        output_dir: Directory to save notebooks
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Create analysis notebook
    analysis_nb = NotebookBuilder.create_template("analysis")
    analysis_nb.save(output_dir / "01_data_analysis.ipynb")

    # Create visualization notebook
    viz_nb = NotebookBuilder.create_template("visualization")
    viz_nb.save(output_dir / "02_visualization.ipynb")

    # Create model notebook
    model_nb = NotebookBuilder.create_template("model")
    model_nb.save(output_dir / "03_ml_model.ipynb")

    # Create custom notebook
    custom_nb = NotebookBuilder("Custom Analysis")
    custom_nb.add_markdown_cell("## Custom Notebook Example")
    custom_nb.add_code_cell("print('This notebook was created programmatically!')")
    custom_nb.add_markdown_cell("### Loading External Data")
    custom_nb.add_code_cell("""# Example of loading external data
from pathlib import Path
import sys
sys.path.append('../')
from src.data_utils import load_external_data

# Load data from parent directory
# data = load_external_data('../../data/mydata.csv')""")
    custom_nb.save(output_dir / "04_custom.ipynb")

    print(f"Created sample notebooks in {output_dir}/")
    return output_dir


if __name__ == "__main__":
    create_sample_notebooks()
