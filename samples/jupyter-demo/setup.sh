#!/bin/bash

# Setup script for Jupyter demo project

echo "Setting up Jupyter demo environment..."

# Install dependencies
echo "Installing dependencies with uv..."
uv sync

# Activate environment
echo "Activating environment..."
source .venv/bin/activate

# Install kernel
echo "Installing Jupyter kernel..."
python -m ipykernel install --user --name=jupyter-demo --display-name="Jupyter Demo"

# Create directories
echo "Creating project directories..."
mkdir -p notebooks data outputs/figures outputs/reports

# Create sample data file
echo "Creating sample data..."
python -c "
import pandas as pd
import numpy as np

np.random.seed(42)
n = 100

data = pd.DataFrame({
    'id': range(1, n+1),
    'value': np.random.randn(n) * 100 + 500,
    'category': np.random.choice(['A', 'B', 'C', 'D'], n),
    'score': np.random.uniform(0, 100, n)
})

data.to_csv('data/sample.csv', index=False)
print('Sample data saved to data/sample.csv')
"

# Create sample notebooks
echo "Creating sample notebooks..."
python src/notebook_creator.py

echo ""
echo "Setup complete! To start working:"
echo "  1. Activate environment: source .venv/bin/activate"
echo "  2. Start Jupyter Lab: jupyter lab"
echo "  3. Open notebooks in the 'notebooks' directory"
echo ""
echo "Available kernels:"
jupyter kernelspec list | grep jupyter-demo