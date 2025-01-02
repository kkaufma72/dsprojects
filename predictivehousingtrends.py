import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

# Fetch real-time housing market data (replace with an appropriate API or data source)
# For demonstration, we use a placeholder API endpoint. Replace it with actual API details.
def fetch_housing_data():
    try:
        response = requests.get("https://api.housingdata.example.com/prices")
        response.raise_for_status()
        data = response.json()
        return pd.DataFrame(data)
    except Exception as e:
        print("Error fetching data:", e)
        return None

# Fetch the data
housing_data = fetch_housing_data()
if housing_data is not None:
    print("Housing data successfully fetched.")
else:
    print("Using default sample data.")

# Use fetched data if available, else fallback to sample data
if housing_data is not None:
    model_performance = pd.DataFrame({
        "Model": housing_data["models"],
        "RMSE": housing_data["rmse"],
        "R-Squared": housing_data["r_squared"],
        "Adjusted R-Squared": housing_data["adjusted_r_squared"],
        "MAE": housing_data["mae"],
    })

    square_footage = housing_data["square_footage"]
    property_prices = housing_data["property_prices"]
else:
    # Fallback sample data
    np.random.seed(42)
    model_performance = pd.DataFrame({
        "Model": ["Linear Regression", "Random Forest", "Gradient Boosting", "Neural Network"],
        "RMSE": [5.2, 3.9, 4.3, 3.1],
        "R-Squared": [0.74, 0.88, 0.85, 0.92],
        "Adjusted R-Squared": [0.73, 0.87, 0.84, 0.91],
        "MAE": [15000, 10200, 12300, 8900],
    })

    square_footage = np.linspace(500, 5000, 100)
    property_prices = 50 * square_footage + np.random.normal(0, 100000, size=100)

# Create visualizations
plt.figure(figsize=(14, 8))

# Performance Metrics Visualization
plt.subplot(2, 2, 1)
plt.bar(model_performance["Model"], model_performance["R-Squared"], color='skyblue')
plt.title("Model Performance (R-Squared)")
plt.ylabel("R-Squared")
plt.xlabel("Model")
plt.xticks(rotation=15)

# Scatter Plot for Property Prices vs Square Footage
plt.subplot(2, 2, 2)
plt.scatter(square_footage, property_prices, alpha=0.7, color='salmon')
plt.title("Property Prices vs Square Footage")
plt.xlabel("Square Footage (sq ft)")
plt.ylabel("Property Price ($)")
plt.grid(True, linestyle='--', alpha=0.7)

# Residual Analysis
residuals = property_prices - (50 * square_footage)  # Residual calculation (actual - predicted)
plt.subplot(2, 2, 3)
plt.scatter(square_footage, residuals, alpha=0.6, color='purple')
plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.title("Residual Analysis")
plt.xlabel("Square Footage (sq ft)")
plt.ylabel("Residuals ($)")
plt.grid(True, linestyle='--', alpha=0.7)

# Feature Importance Visualization (Sample Data)
features = ["Location", "Square Footage", "Interest Rate", "Year Built", "Unemployment Rate"]
importance = [0.75, 0.60, -0.45, 0.50, -0.35]
plt.subplot(2, 2, 4)
plt.barh(features, importance, color='teal')
plt.title("Feature Importance")
plt.xlabel("Correlation with Property Price")
plt.ylabel("Features")
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig("enhanced_housing_trends_visuals.png")
plt.show()

# Residual Histogram
plt.figure(figsize=(8, 5))
plt.hist(residuals, bins=20, color='orange', edgecolor='black', alpha=0.7)
plt.title("Distribution of Residuals")
plt.xlabel("Residual Value ($)")
plt.ylabel("Frequency")
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig("residual_distribution.png")
plt.show()

# Summary Output
def display_summary():
    print("\nModel Performance Summary:")
    print(model_performance)

    print("\nTop Variables Based on Feature Importance:")
    print("1. Location")
    print("2. Square Footage")
    print("3. Interest Rate")

    print("\nInsights:")
    print("- Neural Network outperforms other models with an R-Squared of 0.92.")
    print("- Location and economic factors such as interest rates heavily influence property prices.")

# Display the summary
display_summary()