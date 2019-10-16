############################
# Setup example dataset.
library(SuperLearner)

data <- read.csv2("train_small.csv", header = TRUE, sep = ",")


# Check for any missing data - looks like we don't have any.
# colSums(is.na(Boston))

        
# Extract our outcome variable from the dataframe.
outcome = data$Income

# Create a dataframe to contain our explanatory variables.
data = subset(data, select = -Income)

# Check structure of our dataframe.
str(data)

# Set a seed for reproducibility in this random sampling.
set.seed(1)

# Reduce to a dataset of 150 observations to speed up model fitting.
train_obs = sample(nrow(data), 150)

# X is our training sample.
x_train = data

# Create a holdout set for evaluating model performance.
# Note: cross-validation is even better than a single holdout sample.
x_holdout = data[-train_obs, ]

# Create a binary outcome variable: towns in which median home value is > 22,000.
outcome_bin = as.numeric(outcome > 22)

y_train = outcome_bin[train_obs]
y_holdout = outcome_bin[-train_obs]

# Review the outcome variable distribution.
table(y_train, useNA = "ifany")

###

set.seed(1)
sl = SuperLearner(Y = y_train, X = x_train, family = binomial(),
                  SL.library = c("SL.mean", "SL.glmnet", "SL.ranger"))
sl
sl$times$everything

