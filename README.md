# Movie Rating Prediction Using Machine Learning

## 1. Project Overview

This project predicts the average user rating of a movie using supervised machine learning.
The project treats movie rating prediction as a regression problem because the target value,
`vote_average`, is a continuous numeric value between 0 and 10 in the source dataset.

The project includes the complete machine learning workflow:

1. Load the movie metadata dataset.
2. Inspect the dataset and its statistical properties.
3. Explore the distribution of movie ratings and the relationship between popularity and rating.
4. Clean invalid and missing numeric values.
5. Extract the release year from the release date.
6. Select model features and the target variable.
7. Split the data into training and testing sets.
8. Train several regression models.
9. Evaluate and compare the models.
10. Tune a Random Forest model using grid search.
11. Save the final trained model.
12. Load the saved model to predict the rating of a new movie.

## 2. Machine Learning Task

### Learning type

This is supervised learning. The model learns from historical movie records where the
input features and the actual movie rating are already known.

### Problem type

This is a regression problem. Unlike classification, which predicts categories such as
"good" or "bad", regression predicts a numeric value. Here, the predicted value is the
movie's average rating.

### Target variable

The target variable is:

```text
vote_average
```

This column represents the average rating assigned to a movie. Rows where this value is
missing are excluded from model training because the model needs a known target value to
learn from.

## 3. Dataset

The main input file is:

```text
dataset/movies_metadata.csv
```

The dataset contains movie metadata such as budget, revenue, popularity, runtime, release
date, vote count, and average vote. The repository also contains additional movie-related
CSV files, but the current training script uses `movies_metadata.csv` as its primary source.

### Important source-data considerations

Some columns in the CSV file contain values with inconsistent types. For example, the
`popularity` column can contain numeric strings, scientific notation, missing values, or
the text value `Unknown`. The training script converts the selected numeric columns with
`pd.to_numeric(..., errors="coerce")`. Invalid values become missing numeric values and are
then replaced with the median of the relevant feature column.

## 4. Features

The model uses the following six input features:

| Feature | Description |
| --- | --- |
| `budget` | The production budget of the movie. |
| `popularity` | A popularity score from the source dataset. |
| `revenue` | The movie's reported revenue. |
| `runtime` | The movie duration in minutes. |
| `vote_count` | The number of votes received by the movie. |
| `release_year` | The year extracted from `release_date`. |

The original `release_date` column is converted to a datetime value. A new numeric
`release_year` feature is extracted because machine learning models work more directly
with a numeric year than with a complete date string.

## 5. Data Cleaning

The cleaning process in `train_model.py` performs these operations:

### Numeric conversion

The columns `budget`, `popularity`, `revenue`, `runtime`, `vote_average`, and `vote_count`
are converted to numeric values. Values that cannot be parsed are converted to `NaN`.

### Duplicate removal

Duplicate movie records are removed before model training. This prevents repeated records
from giving some examples more influence than others.

### Missing feature values

Missing values in the numeric input features are replaced with the median value of their
respective column. Median imputation is less affected by extreme values than mean
imputation, which is useful for fields such as revenue and vote count.

### Missing text values

Missing text values are filled with `Unknown` so that the exploratory analysis has a
consistent representation for missing text fields. These text columns are not passed to
the current numeric machine learning models.

### Missing target values

Missing `vote_average` values are not imputed as training targets. Instead, those rows are
removed from the modeling dataset because inventing a target rating could introduce noise
into the supervised learning process.

## 6. Exploratory Data Analysis

The training script creates two visualizations in `outputs/eda/`:

### Rating distribution

```text
outputs/eda/rating_distribution.png
```

This histogram shows how the movie ratings are distributed across the dataset. It helps
identify whether ratings are concentrated around a particular range and whether unusual
values or a skewed distribution may affect model performance.

### Popularity versus rating

```text
outputs/eda/popularity_vs_rating.png
```

This scatter plot compares movie popularity with average rating. It provides a visual
starting point for understanding whether popular movies tend to receive higher or lower
ratings, although a plot alone cannot establish causation.

## 7. Models

The project trains four regression models.

### Linear Regression

Linear Regression estimates the target as a weighted combination of the input features.
It is simple and easy to interpret, and it provides a useful baseline. However, movie
ratings may not have a purely linear relationship with budget, popularity, revenue, or
vote count.

### Decision Tree Regressor

A Decision Tree splits the data into smaller groups using feature conditions. It can model
nonlinear relationships, but a single tree can overfit the training data.

### Random Forest Regressor

Random Forest combines many decision trees and averages their predictions. The averaging
usually improves stability and reduces the risk of relying on one tree's particular
splits.

### Gradient Boosting Regressor

Gradient Boosting builds trees sequentially. Each new tree focuses on correcting errors
made by the previous trees. This allows it to model complex relationships, but it can be
more sensitive to configuration choices.

## 8. Evaluation Metrics

The project reports four metrics in `outputs/model_results.csv`.

### Mean Absolute Error (MAE)

MAE is the average absolute difference between the actual and predicted ratings. An MAE
of 0.8 means that predictions are off by about 0.8 rating points on average.

Lower MAE is better.

### Mean Squared Error (MSE)

MSE averages the squared prediction errors. Because errors are squared, larger mistakes
receive more penalty than smaller mistakes.

Lower MSE is better.

### Root Mean Squared Error (RMSE)

RMSE is the square root of MSE. It is expressed in the same units as the target rating,
which makes it easier to interpret than MSE.

Lower RMSE is better. The project uses RMSE to select the initial best model.

### R2 score

R2 measures how much of the variation in the target is explained by the model relative
to a baseline that always predicts the average rating.

Higher R2 is better. An R2 value can be negative when a model performs worse than that
simple average-rating baseline.

## 9. Model Comparison and Tuning

After the four initial models are evaluated, their metrics are placed in a comparison
table and saved here:

```text
outputs/model_results.csv
```

The script selects the initial best model by finding the row with the smallest RMSE.
The project then performs hyperparameter tuning for a Random Forest using `GridSearchCV`.

The search tests these combinations:

| Parameter | Values tested |
| --- | --- |
| `n_estimators` | 100, 200 |
| `max_depth` | 10, 20, None |
| `min_samples_split` | 2, 5 |
| Cross-validation | 3 folds |

The search uses negative mean squared error as the scikit-learn scoring value. Scikit-learn
uses a negative value because its model-selection interface maximizes scores, while lower
mean squared error is better. The best estimator is then tested on the held-out test set
using the same evaluation function.

## 10. Project Structure

```text
Movie_Rating_Prediction/
|
|-- dataset/
|   |-- movies_metadata.csv
|   |-- credits.csv
|   |-- keywords.csv
|   |-- links.csv
|   |-- links_small.csv
|   |-- ratings.csv
|   `-- ratings_small.csv
|
|-- models/
|   `-- movie_rating_model.pkl
|
|-- outputs/
|   |-- eda/
|   |   |-- rating_distribution.png
|   |   `-- popularity_vs_rating.png
|   `-- model_results.csv
|
|-- train_model.py
|-- predict.py
|-- requirements.txt
`-- README.md
```

### File descriptions

#### `train_model.py`

Loads and cleans the dataset, creates the features, trains the models, evaluates them,
tunes the Random Forest, and saves the final model.

#### `predict.py`

Loads the saved model and creates a prediction from one new movie record. It does not
retrain the model.

#### `requirements.txt`

Lists the Python libraries required by the project.

#### `models/movie_rating_model.pkl`

Contains the serialized tuned model created by `joblib.dump`. This file is loaded by
`predict.py`.

#### `outputs/model_results.csv`

Contains the MAE, MSE, RMSE, and R2 values for the four initial models.

## 11. Installation

Use Python 3.10 or newer where possible. Create and activate a virtual environment to keep
the project dependencies separate from other Python projects.

### Create a virtual environment on Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell prevents activation because of the execution policy, use the Python
interpreter inside the environment directly or consult your system's Python policy.

### Install dependencies

```bash
pip install -r requirements.txt
```

The required libraries are:

- `pandas` for loading and manipulating tabular data.
- `numpy` for numeric operations such as calculating RMSE.
- `matplotlib` and `seaborn` for exploratory plots.
- `scikit-learn` for splitting data, training models, tuning parameters, and calculating metrics.
- `joblib` for saving and loading the trained model.

## 12. Running the Project

Run commands from the project root, the folder containing `train_model.py`.

### Train the models

```bash
python train_model.py
```

The script prints dataset information, model metrics, the selected model, the best Random
Forest parameters, and the final tuned-model metrics. It also creates or updates:

```text
models/movie_rating_model.pkl
outputs/model_results.csv
outputs/eda/rating_distribution.png
outputs/eda/popularity_vs_rating.png
```

The hyperparameter search can take longer than the initial model training because it fits
multiple Random Forest configurations using three-fold cross-validation.

### Predict a rating

After training has created the model file, run:

```bash
python predict.py
```

The script loads `models/movie_rating_model.pkl`, creates a sample movie record, and prints
the predicted rating.

## 13. Making a Prediction for Another Movie

Open `predict.py` and replace the values in the `new_movie` DataFrame. The column names
must remain exactly the same as the training feature names:

```python
new_movie = pd.DataFrame({
	"budget": [10000000],
	"popularity": [50],
	"revenue": [50000000],
	"runtime": [120],
	"vote_count": [500],
	"release_year": [2026]
})
```

The values should be numeric. Use the same units as the source dataset: budget and revenue
should use the dataset's currency units, runtime should be in minutes, vote count should
be a count, and release year should be a four-digit year.

The prediction is an estimate based on patterns learned from historical data. It should
not be treated as a guaranteed review score or as a replacement for real audience feedback.

## 14. Reproducibility

The train/test split and tree-based models use `random_state=42`, which makes the results
repeatable when the same data, Python packages, and hardware behavior are used. Results
can still vary if the dataset or dependency versions change.

The test set is kept separate from model fitting and is used to estimate performance on
unseen records. The grid search uses only the training data during cross-validation, then
the selected estimator is evaluated on the held-out test data.

## 15. Troubleshooting

### File not found errors

Run commands from the project root. Relative paths such as `dataset/movies_metadata.csv`
and `models/movie_rating_model.pkl` are resolved from the current working directory.

### Missing model file

Run the training script before the prediction script:

```bash
python train_model.py
python predict.py
```

### Dependency errors

Install the listed packages again:

```bash
pip install -r requirements.txt
```

### Plot display warnings

The training script saves the plots before calling `plt.show()`. In a headless environment,
Matplotlib may warn that the figure cannot be displayed even though the image files are
successfully written to `outputs/eda/`.

### CSV type warnings

The source dataset contains mixed values in some columns. The script intentionally converts
the selected model columns with `errors="coerce"` before filling missing feature values.
This allows values such as scientific notation in the `popularity` column to be used safely.

## 16. Limitations and Possible Improvements

The current model uses six numeric features and does not use text, genre, cast, crew,
language, or production-company information. These fields may contain useful predictive
signals but would require additional feature engineering.

Other possible improvements include:

- Adding genre or language features with one-hot encoding.
- Applying logarithmic transformations to highly skewed budget, revenue, and vote-count values.
- Testing additional regressors and stronger baselines.
- Using a pipeline to keep preprocessing and model prediction together.
- Comparing cross-validation results across all model families.
- Checking for data leakage and evaluating performance on a newer time-based test set.
- Saving the preprocessing rules along with the model when the feature pipeline becomes more complex.

## 17. License and Dataset Attribution

Add the appropriate dataset attribution and license information before publishing or
redistributing the project. The repository should follow the terms that apply to the
original movie metadata and ratings data.
