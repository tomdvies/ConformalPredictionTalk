import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import matplotlib

matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"
matplotlib.rcParams.update({"font.size": 12})
# np.random.seed(42)


def estimate_sigma(x, residuals, x_eval, window_size=1.0):
    """Estimate local standard deviation using sliding window average

    Args:
        x: Input locations where we have residuals
        residuals: Squared residuals from the fit
        x_eval: Points where we want to estimate sigma
        window_size: Size of local window for averaging

    Returns:
        sigma: Estimated standard deviation at each x_eval point
    """
    sigma = np.zeros(len(x_eval))

    for i, xi in enumerate(x_eval):
        # Find points within the window
        mask = np.abs(x - xi) <= window_size

        # Need at least 3 points to estimate variance
        if np.sum(mask) < 3:
            window_size_i = window_size
            while np.sum(mask) < 3:
                window_size_i *= 1.5
                mask = np.abs(x - xi) <= window_size_i

        # Local average of squared residuals
        sigma[i] = np.sqrt(np.mean(residuals[mask]))

    return sigma


def estimate_sigma_smooth(x, residuals, x_eval, bandwidth=1.0):
    """Estimate local standard deviation using Gaussian kernel smoothing

    Args:
        x: Input locations where we have residuals
        residuals: Squared residuals from the fit
        x_eval: Points where we want to estimate sigma
        bandwidth: Kernel bandwidth parameter

    Returns:
        sigma: Estimated standard deviation at each x_eval point
    """
    sigma = np.zeros(len(x_eval))

    for i, xi in enumerate(x_eval):
        # Gaussian kernel weights
        weights = np.exp(-0.5 * ((x - xi) / bandwidth) ** 2)
        weights = weights / np.sum(weights)  # Normalize weights

        # Weighted average of squared residuals
        sigma[i] = np.sqrt(np.sum(weights * residuals))

    return sigma


def get_conformal_predictor(x, y, degree=3):
    """
    Implements locally-weighted split conformal prediction for polynomial regression.

    Args:
        x: Input features
        y: Target values
        degree: Degree of polynomial fit

    Returns:
        coeffs: Polynomial coefficients
        confinterval: Function that generates prediction intervals
    """
    x_train, x_calib, y_train, y_calib = train_test_split(
        x, y, test_size=0.3, random_state=42
    )

    # Fit polynomial regression
    coeffs = np.polyfit(x_train, y_train, degree)

    # Get predictions on calibration set
    y_pred_calib = np.polyval(coeffs, x_calib)
    raw_residuals = y_calib - y_pred_calib

    # Estimate local standard deviation
    squared_residuals = raw_residuals**2
    sigma_calib = estimate_sigma_smooth(x_calib, squared_residuals, x_calib)

    # Compute studentized residuals
    studentized_residuals = np.abs(raw_residuals / sigma_calib)

    def confinterval(x_new, alpha=0.9):
        """Generate prediction intervals for new x values."""
        y_pred = np.polyval(coeffs, x_new)

        # Estimate sigma at new points
        sigma_new = estimate_sigma_smooth(x_calib, squared_residuals, x_new)

        # Get quantile of studentized residuals
        q = np.quantile(studentized_residuals, alpha)

        # Scale intervals by local sigma
        return y_pred - q * sigma_new, y_pred + q * sigma_new

    return coeffs, confinterval


def mean_function(x):
    return 1 + (1 / 6) * x**2 + (1 / 24) * x**3


def sample_data(n_samples, xstart=-5, xend=5, noisevar=0.2):
    x = np.random.uniform(xstart, xend, n_samples)
    noise = np.random.normal(0, noisevar * (1 + np.abs(x) ** 2), x.shape)
    y = mean_function(x) + noise
    return x, y


# Generate synthetic data
xstart = -5
xend = 5
noisevar = 0.2

x, y = sample_data(400, xstart, xend, noisevar)
x_sorted = np.sort(x)

# Fit model and generate prediction grid
coeffs, conf_pred = get_conformal_predictor(x, y, degree=3)
# x_fine = np.linspace(x_sorted.min(), x_sorted.max(), 1000)
x_fine = np.linspace(xstart, xend, 1000)
y_fit = np.polyval(coeffs, x_fine)

# Visualization
plt.figure(figsize=(10, 6))
plt.scatter(x, y, alpha=0.5, label="Data points")
plt.plot(x_fine, mean_function(x_fine), "r-", label="Mean Cubic Function")
plt.plot(x_fine, y_fit, "g--", label="Fitted curve")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Cubic Fit With Heteroskedastic Variance")
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
x_test, y_test = sample_data(1000, xstart, xend, noisevar)  # Generate fresh test data
# Compute empirical coverage on fresh test set
lower, upper = conf_pred(x_test)
coverage = np.mean((y_test >= lower) & (y_test <= upper))
print(f"\nEmpirical coverage on fresh test set: {coverage:.3f}")
print(f"Target coverage: 0.9")

# Demonstrate prediction intervals
# x_example = np.array([0.5])
# lower, upper = conf_pred(x_example)
# print("\nPrediction intervals for x = f(y) +-", (upper - lower)[0])
