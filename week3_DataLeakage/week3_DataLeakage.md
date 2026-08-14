# Week 3: Data Preprocessing Pipelines and Leakage Detection on the Titanic Dataset

This project demonstrates how data leakage can enter a machine-learning workflow through preprocessing and feature engineering. Using the Titanic dataset and logistic regression, the notebook compares deliberately flawed workflows with a leakage-free scikit-learn pipeline and shows why a high test score is not trustworthy when held-out information influences training.

## Notebook

`week3_DataLeakage.ipynb`

## Objectives

- Identify where preprocessing can leak test-set information into training.
- Compare preprocessing before and after the train-test split.
- Build a leakage-free pipeline for imputation, scaling, encoding, and classification.
- Examine leakage caused by full-dataset imputation.
- Demonstrate severe target leakage through ticket-based target encoding.
- Compare held-out accuracy and F1 scores across flawed and corrected workflows.
- Summarize practical rules for preventing data leakage.

## Dataset

The notebook expects `Titanic-Dataset.xls` in the same directory. Despite its extension, the file is read as CSV-formatted text with `pandas.read_csv()`.

The dataset contains 891 passengers. The target is `Survived`, whose class distribution is:

- Did not survive (`0`): 61.62%
- Survived (`1`): 38.38%

Seven input features are selected:

1. `Pclass`
2. `Age`
3. `SibSp`
4. `Parch`
5. `Fare`
6. `Sex`
7. `Embarked`

`PassengerId`, `Name`, `Ticket`, and `Cabin` are excluded from the baseline model to avoid high-cardinality identifiers and keep the pipeline interpretable. `Ticket` is used later only to demonstrate target leakage.

The raw observations are split into 70% training data and 30% testing data with stratification and `random_state=67`.

## Workflow

### 1. Flawed preprocessing before splitting

The first workflow fits all preprocessing steps on the complete feature matrix before creating the train and test sets. The preprocessing includes:

- Median imputation and standardization for numeric features
- Most-frequent imputation and one-hot encoding for categorical features

Although the target labels are not used, the imputer, scaler, and encoder learn from the held-out feature distribution. This makes the test set no longer independent.

### 2. Corrected leakage-free pipeline

The corrected workflow splits the raw data first and combines preprocessing with `LogisticRegression` in a scikit-learn `Pipeline`. Calling `fit()` only on the training partition ensures that every imputation value, scaling statistic, and encoding category is learned without access to test rows.

The notebook illustrates that the learned feature statistics differ between the complete dataset and the training partition:

| Statistic | Full-data value | Training-only value | Difference |
| --- | ---: | ---: | ---: |
| Age mean | 29.6991 | 29.8468 | -0.1477 |
| Fare mean | 32.2042 | 30.7527 | 1.4515 |

### 3. Full-data imputation leakage

A second flawed workflow calculates missing-value replacements from the complete dataset before splitting. For this particular split, the full-data and training-only medians and mode happen to be identical:

| Feature | Full-data value | Training-only value |
| --- | ---: | ---: |
| Age | 28.0 | 28.0 |
| Fare | 14.4542 | 14.4542 |
| Embarked | S | S |

This equality does not make the method valid. The full-data calculation still gives the training process access to information that would be unavailable for genuinely unseen observations.

### 4. Ticket-based target leakage

The most severe flawed workflow creates `TicketSurvivalRate_LEAK` by grouping the complete dataset by `Ticket` and calculating each ticket group's mean `Survived` value before the split.

Of the 891 passengers, 547 have a ticket value that occurs only once. For those passengers, the engineered feature is exactly equal to their own target label. Shared-ticket values also include outcomes from the passenger and potentially from held-out rows, producing an unrealistically powerful predictor.

### 5. Metric comparison

| Workflow | Accuracy | F1 score |
| --- | ---: | ---: |
| All preprocessing before split | 80.60% | 73.74% |
| Full-data imputation before split | 80.60% | 73.74% |
| Full-data ticket target encoding | 98.51% | 98.10% |
| Corrected leakage-free pipeline | 80.60% | 73.74% |

The first two flawed workflows happen to produce the same rounded metrics as the corrected pipeline. Leakage does not always increase a score in a particular sample; the problem is that the evaluation is contaminated and cannot be treated as a reliable estimate of performance on unseen data. The ticket-based target feature demonstrates how severe leakage can produce dramatically inflated results.

## Requirements

- Python 3.9 or later
- Jupyter Notebook or JupyterLab
- NumPy
- pandas
- scikit-learn

Install the dependencies with:

```bash
python -m pip install jupyter numpy pandas scikit-learn
```

## Running the Project

Place `Titanic-Dataset.xls` in this directory, then start Jupyter from the directory:

```bash
jupyter notebook week3_DataLeakage.ipynb
```

Run the notebook cells in order.

> **Dataset note:** `Titanic-Dataset.xls` must contain CSV-formatted text because the notebook loads it with `pd.read_csv()`. The dataset file is not included in the current directory.

## Key Observations

- Any data-dependent transformation fitted before splitting can leak information, even when it does not use the target directly.
- Equal scores do not prove that a leaky workflow is safe; they only show that the contamination did not change these rounded metrics for this split.
- Target-derived features can reveal labels directly and make held-out performance appear nearly perfect.
- Imputation, encoding, scaling, feature selection, and model fitting should be placed in a pipeline fitted only on training data.
- During development, preprocessing must be refitted within each cross-validation fold, and the test set should be reserved for one final evaluation.
- Target encoding, when appropriate, must generate training encodings out of fold and derive validation or test mappings only from training labels.
