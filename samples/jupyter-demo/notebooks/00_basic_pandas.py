# %% [markdown]
# # My First Analysis
# Welcome to reproducible data science!

# %%
import pandas as pd

# Generate sample data
data = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'score': [85, 92, 78]
})
print(data)

# %%
# Quick visualization
data.plot(x='name', y='score', kind='bar', title='Scores')
# %%
