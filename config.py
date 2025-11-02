# config.py
# Holds all configuration for the AutoML pipeline

# --- File Paths ---
# Create a 'data' folder in the same directory and put your CSV there
DATA_PATH = "data/ag_news_train.csv" 

# --- Target Variable ---
TARGET_COLUMN = "label" # Change this to your target variable name

# --- Model Settings ---
# Define problem type: 'classification' or 'regression'
PROBLEM_TYPE = "classification" 

# --- Preprocessing Steps ---
# Toggle which steps to apply
APPLY_IMPUTATION = True
APPLY_SCALING = True
APPLY_ENCODING = True


# These rules decide what "type" each column is.

# TEXT_AVG_WORD_THRESHOLD
# If a column's average cell has this many words or more, it's 'text'.
# 5 is a good start. 'New York' is 2 words. 'A short product review' is 4.
# Anything averaging 5+ is almost certainly free-form text.
TEXT_AVG_WORD_THRESHOLD = 5

# HIGH_CARD_THRESHOLD
# If a non-text column has *more* unique values than this, it's 'high_cardinality'.
# We use this to avoid OneHotEncoding a column with 10,000 unique zip codes,
# which would create 10,000 new features.
# 50 is a common-sense default.
HIGH_CARD_THRESHOLD = 50


# TEXT_MAX_FEATURES
# How many columns (i.e., top words) to create from TF-IDF.
# This is vital for controlling the dimensionality of your dataset.
# 1000 is a good starting point.
TEXT_MAX_FEATURES = 1000

# TEXT_NGRAM_RANGE
# The range of "n-grams" to capture.
# (1, 1) means only single words ("New", "York").
# (1, 2) means single words AND two-word pairs ("New", "York", "New York").
# (1, 2) can capture more meaning but creates *many* more features.
# Start with (1, 1).
TEXT_NGRAM_RANGE = (1, 1)

NUMERIC_SCALER = "standard" # or "minmax" or "robust"

# ID_COLUMNS = ['match id', 'team1', 'team1_id', 'team1_roster_ids', 'team2', 'team2_id', 'team2_roster_ids','winner', 'winner_id', 'win_1_2', 'toss winner', 'toss decision']
ID_COLUMNS = []

# --- Feature Engineering Settings ---
APPLY_FEATURE_ENGINEERING = True

# Drop columns with missing values > 95%
FE_HIGH_NULL_THRESHOLD = 0.95

# Drop columns with Pearson correlation > 0.90
FE_CORR_THRESHOLD = 0.90

# Use Lasso (L1) to select features (runs before PCA, if both are True)
FE_USE_LASSO = True

# Use PCA for dimensionality reduction (will disable Lasso)
# Note: This is often not needed for tree-models, but good for linear models.
FE_USE_PCA = False
FE_PCA_EXPLAINED_VAR = 0.95

# --- Model Tuning ---
TUNING_STRATEGY = "GridSearchCV" # or "RandomizedSearchCV"
CV_FOLDS = 5

