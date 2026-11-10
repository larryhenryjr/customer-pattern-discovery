# Customer Pattern Discovery

A compact unsupervised-learning toolkit for turning complex customer behavior into interpretable segments, principal components, and cross-sell associations.

## What this project demonstrates

- **K-means segmentation:** groups customers by scaled behavioral and service variables, then returns centroids in the original business units.
- **Principal component analysis:** reduces correlated feature spaces while retaining a configurable share of variance.
- **Association-rule mining:** discovers frequently co-occurring products and ranks relationships by support, confidence, and lift.

These three techniques answer different questions: who behaves similarly, which dimensions explain the variation, and which products tend to move together.

## Portfolio artifacts

- [Rendered analysis notebook](notebook/analysis.html)
- [Portfolio report](reports/customer-pattern-discovery-report.pdf)

The public repository includes the rendered notebook and report while intentionally excluding the working `.ipynb` file, restricted source data, local paths, and academic grading material.
## Repository structure

```text
├── data/README.md
├── src/patterns.py
├── tests/test_patterns.py
├── requirements.txt
└── README.md
```

## Run locally

```bash
python -m venv .venv
pip install -r requirements.txt
pytest tests/ -v --tb=short
```

Import the functions into a notebook, Databricks workflow, or Python application:

```python
from src.patterns import mine_associations, reduce_dimensions, segment_customers
```

## Design principles

- Validate inputs before modeling.
- Scale continuous features before distance-based clustering or PCA.
- Return cluster centroids in original units so stakeholders can interpret them.
- Keep transaction logic separate from customer-level feature modeling.
- Treat discovered patterns as decision signals, not causal relationships.

## Portfolio

Case study: [larry-henry-jr.net](https://larry-henry-jr.net/#project/customer-pattern-discovery)
