# Machine Learning Lab - Semester 5

This repository contains my Semester 5 Machine Learning Laboratory coursework. It is organized by week and includes Jupyter notebooks, implementations of machine learning algorithms, data exploration, visualizations, experiments, and observations from each lab exercise.

## Course Objectives

- Understand the foundations of machine learning.
- Implement and evaluate supervised and unsupervised learning algorithms.
- Explore datasets through statistical analysis and visualization.
- Compare models using appropriate performance metrics.
- Study the effects of preprocessing, feature scaling, and hyperparameters.
- Develop practical experience with Python's machine learning ecosystem.

## Repository Structure

```text
MachineLearningLab_Sem5/
|-- README.md
`-- week1_KNN/
    |-- MachineLearningLabAssignment1_KrishBansal_24CS248.ipynb
    `-- week1_KNN.md
```

Additional weekly exercises will be added as the course progresses.

## Lab Work

| Week | Topic | Description | Documentation |
| --- | --- | --- | --- |
| 1 | K-Nearest Neighbors | Iris classification, exploratory visualization, feature scaling, model evaluation, class probabilities, and decision boundaries | [Week 1 README](week1_KNN/week1_KNN.md) |

## Technologies Used

- Python
- Jupyter Notebook
- NumPy
- pandas
- Matplotlib
- Seaborn
- scikit-learn

The exact dependencies may vary between weekly exercises.

## Getting Started

### 1. Clone the repository

```bash
git clone <repository-url>
cd MachineLearningLab_Sem5
```

Replace `<repository-url>` with the Git URL of this repository.

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows, activate it with:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install common dependencies

```bash
python -m pip install jupyter numpy pandas matplotlib seaborn scikit-learn
```

### 4. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open the relevant weekly directory and run its notebook cells in order. Refer to the documentation inside each folder for exercise-specific instructions and observations.

## Current Work

### Week 1: K-Nearest Neighbors

The first lab uses the Iris dataset to train and compare KNN classifiers. It evaluates different values of `k`, compares scaled and unscaled data, calculates class probabilities, and plots decision boundaries using uniform and distance-based neighbor weighting.

See [week1_KNN.md](week1_KNN/week1_KNN.md) for complete details and recorded results.

## Notes

- Notebook outputs may differ slightly when random seeds, library versions, or model parameters change.
- Each weekly folder is intended to remain self-contained.
- Lab-specific setup details and findings are documented alongside the corresponding notebook.
