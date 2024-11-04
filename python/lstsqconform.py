import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import matplotlib

matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"
matplotlib.rcParams.update({"font.size": 12})
# np.random.seed(42)


def get_conformal_predictor(x, y, degree=3):
    """
    Implements split conformal prediction for polynomial regression.

    Args:
        x: Input features
        y: Target values
        degree: Degree of polynomial fit

    Returns:
        coeffs: Polynomial coefficients
        confinterval: Function that generates prediction intervals
    """
    x_train, x_calib, y_train, y_calib = train_test_split(
        x, y, test_size=0.5, random_state=42
    )

    coeffs = np.polyfit(x_train, y_train, degree)

    y_pred_calib = np.polyval(coeffs, x_calib)
    residuals = np.abs(y_calib - y_pred_calib)

    def confinterval(x_new, alpha=0.9):
        """Generate prediction intervals for new x values."""
        y_pred = np.polyval(coeffs, x_new)
        q = np.quantile(residuals, alpha)
        return y_pred - q, y_pred + q

    return coeffs, confinterval


# Generate synthetic data
xstart = -5
xend = 5
noisevar = 0.2

x = np.random.uniform(xstart, xend, 100)
x_sorted = np.sort(x)


def true_function(x):
    """Ground truth function: f(x) = 1 + x + x^2"""
    return 1 + (1 / 6) * x**2 + (1 / 24) * x**3


# Add heteroscedastic noise
noise = np.random.normal(0, noisevar, x.shape)
y = true_function(x) + noise

# Fit model and generate prediction grid
coeffs, conf_pred = get_conformal_predictor(x, y, degree=5)
# x_fine = np.linspace(x_sorted.min(), x_sorted.max(), 1000)
x_fine = np.linspace(xstart, xend, 1000)
y_fit = np.polyval(coeffs, x_fine)

# Visualization
plt.figure(figsize=(10, 6))
plt.scatter(x, y, alpha=0.5, label="Data points")
plt.plot(x_fine, true_function(x_fine), "r-", label="True Cubic Function")
plt.plot(x_fine, y_fit, "g--", label="Fitted curve")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Cubic Fit")
plt.grid()

# Generate and plot prediction intervals
lower_fine, upper_fine = conf_pred(x_fine)
plt.fill_between(
    x_fine,
    lower_fine,
    upper_fine,
    alpha=0.2,
    color="green",
    label="90% Prediction Interval",
)
plt.legend(framealpha=1)
plt.show()

# Evaluate model performance
print("Fitted coefficients (highest power first):")
poly_terms = []
for i, coef in enumerate(coeffs):
    if abs(coef) < 1e-10:  # Skip terms that are effectively zero
        continue
    power = len(coeffs) - i - 1
    if power == 0:
        term = f"{coef:.3f}"
    elif power == 1:
        term = f"{coef:.3f}x"
    else:
        term = f"{coef:.3f}x^{power}"
    poly_terms.append(term)
print("y = " + " + ".join(poly_terms).replace("+ -", "- "))
# Generate fresh test data for coverage evaluation
x_test = np.random.uniform(xstart, xend, 1000)  # Same distribution as training data
# x_test = np.random.normal(0, xvar, 1000)  # Same distribution as training data
test_noise = np.random.normal(0, noisevar, x_test.shape)
y_test = true_function(x_test) + test_noise

# Compute empirical coverage on fresh test set
lower, upper = conf_pred(x_test)
coverage = np.mean((y_test >= lower) & (y_test <= upper))
print(f"\nEmpirical coverage on fresh test set: {coverage:.3f}")
print(f"Target coverage: 0.9")

# Demonstrate prediction intervals
x_example = np.array([0.5])
lower, upper = conf_pred(x_example)
print("\nPrediction intervals for x = f(y) +-", (upper - lower)[0])
