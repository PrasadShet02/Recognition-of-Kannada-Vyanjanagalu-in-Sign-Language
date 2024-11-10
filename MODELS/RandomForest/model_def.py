from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Create a Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# You can experiment with different hyperparameters as needed
