# Mixed Effects Modeling in R
# This script demonstrates how to fit and analyze mixed effects models using lme4

# Install required packages if not already installed
if (!require("lme4")) install.packages("lme4")
if (!require("lmerTest")) install.packages("lmerTest")
if (!require("ggplot2")) install.packages("ggplot2")

# Load required packages
library(lme4)
library(lmerTest)
library(ggplot2)

# Example 1: Simple Linear Mixed Effects Model
# Let's create some example data
set.seed(123)
n_subjects <- 20
n_observations <- 5
n_total <- n_subjects * n_observations

# Create example data frame
data <- data.frame(
  subject = factor(rep(1:n_subjects, each = n_observations)),
  time = rep(1:n_observations, times = n_subjects),
  treatment = factor(rep(c("A", "B"), each = n_total/2)),
  response = rnorm(n_total, mean = 10, sd = 2)
)

# Add some random effects
data$response <- data$response + rnorm(n_subjects, sd = 1)[data$subject]

# Fit a linear mixed effects model
# This model includes:
# - Fixed effects: time and treatment
# - Random effects: random intercepts for subjects
model1 <- lmer(response ~ time + treatment + (1|subject), data = data)

# View model summary
summary(model1)

# Get confidence intervals
confint(model1)

# Example 2: More Complex Mixed Effects Model
# Let's create another example with nested random effects
set.seed(456)
n_schools <- 10
n_classes_per_school <- 3
n_students_per_class <- 5

# Create nested data
school_data <- data.frame(
  school = factor(rep(1:n_schools, each = n_classes_per_school * n_students_per_class)),
  class = factor(rep(1:(n_schools * n_classes_per_school), each = n_students_per_class)),
  student = factor(1:(n_schools * n_classes_per_school * n_students_per_class)),
  pretest = rnorm(n_schools * n_classes_per_school * n_students_per_class, mean = 70, sd = 10),
  posttest = rnorm(n_schools * n_classes_per_school * n_students_per_class, mean = 75, sd = 10)
)

# Add some random effects
school_data$posttest <- school_data$posttest + 
  rnorm(n_schools, sd = 2)[school_data$school] +
  rnorm(n_schools * n_classes_per_school, sd = 1)[school_data$class]

# Fit a nested mixed effects model
# This model includes:
# - Fixed effects: pretest score
# - Random effects: random intercepts for schools and classes within schools
model2 <- lmer(posttest ~ pretest + (1|school/class), data = school_data)

# View model summary
summary(model2)

# Get confidence intervals
confint(model2)

# Example 3: Visualization of Mixed Effects
# Create a plot to visualize the random effects
ggplot(data, aes(x = time, y = response, color = treatment, group = subject)) +
  geom_line(alpha = 0.5) +
  geom_point() +
  theme_minimal() +
  labs(title = "Response over Time by Treatment and Subject",
       x = "Time",
       y = "Response",
       color = "Treatment")

# Example 4: Model Comparison
# Fit a simpler model without random effects
model_simple <- lm(response ~ time + treatment, data = data)

# Compare models using AIC
AIC(model1, model_simple)

# Example 5: Diagnostics
# Check residuals
par(mfrow = c(2, 2))
plot(model1)

# Check random effects
ranef(model1)

# Example 6: Prediction
# Create new data for prediction
new_data <- expand.grid(
  time = 1:5,
  treatment = c("A", "B"),
  subject = factor(1)
)

# Make predictions
predictions <- predict(model1, newdata = new_data, re.form = NA)

# Add predictions to new data
new_data$predicted <- predictions

# Plot predictions
ggplot(new_data, aes(x = time, y = predicted, color = treatment)) +
  geom_line() +
  geom_point() +
  theme_minimal() +
  labs(title = "Predicted Response over Time by Treatment",
       x = "Time",
       y = "Predicted Response",
       color = "Treatment") 