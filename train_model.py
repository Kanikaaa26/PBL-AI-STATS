from app.preprocess import load_and_preprocess_metadata
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ✅ Feature names from .names file (32 features)
feature_names = [
    "erythema", "scaling", "definite borders", "itching", "koebner phenomenon", 
    "polygonal papules", "follicular papules", "oral mucosal involvement", 
    "knee and elbow involvement", "scalp involvement", "family history", 
    "melanin incontinence", "eosinophils in the infiltrate", "PNL infiltrate", 
    "fibrosis of the papillary dermis", "exocytosis", "acanthosis", "hyperkeratosis", 
    "parakeratosis", "clubbing of the rete ridges", "elongation of the rete ridges", 
    "thinning of the suprapapillary epidermis", "spongiform pustule", 
    "munro microabcess", "focal hypergranulosis", "disappearance of the granular layer", 
    "vacuolisation and damage of basal layer", "spongiosis", 
    "saw-tooth appearance of retes", "follicular horn plug", 
    "perifollicular parakeratosis", "inflammatory monoluclear infiltrate"
]

# ✅ Load and preprocess data
X, y = load_and_preprocess_metadata('data/dermatology.csv')
print("Shape of features:", X.shape)
print("Shape of labels:", y.shape)

# ✅ Balance dataset using RandomOverSampler
ros = RandomOverSampler(random_state=42)
X, y = ros.fit_resample(X, y)
print("Shape of balanced features:", X.shape)
print("Shape of balanced labels:", y.shape)

# ✅ Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ✅ Convert to DataFrames with feature names
X_train_df = pd.DataFrame(X_train, columns=feature_names)
X_test_df = pd.DataFrame(X_test, columns=feature_names)

# ✅ Train model
model = RandomForestClassifier(random_state=42, class_weight='balanced')
model.fit(X_train_df, y_train)

# ✅ Evaluate model
y_pred = model.predict(X_test_df)
print("\nModel Evaluation:")
print(classification_report(y_test, y_pred, zero_division=1))

# ✅ SHAP Explainability
explainer = shap.TreeExplainer(model, X_train_df)
shap_values = explainer.shap_values(X_test_df, check_additivity=False)
shap_values = np.array(shap_values)  # Ensure NumPy array

print("Actual full SHAP shape:", shap_values.shape)  # Should be (232, 32, 4)

# ✅ Loop over all classes
shap.initjs()
num_classes = shap_values.shape[2]  # shape = (n_samples, n_features, n_classes)

for class_idx in range(num_classes):
    print(f"\n📊 SHAP Summary for Class {class_idx}")
    
    # Correctly slice SHAP values for the class
    shap_values_class = shap_values[:, :, class_idx]  # (samples, features)

    # ✅ Summary Plot (Bar)
    plt.figure()
    shap.summary_plot(shap_values_class, X_test_df, plot_type="bar", show=False)
    plt.title(f"SHAP Summary Plot - Class {class_idx}")
    plt.tight_layout()
    plt.savefig(f"shap_summary_class_{class_idx}.png")
    plt.close()

    # ✅ Force Plot (for first instance)
    force_plot = shap.force_plot(
    explainer.expected_value[class_idx],
    shap_values_class[0],
    X_test_df.iloc[0],
    matplotlib=False  
)

# Save interactive HTML
shap.save_html(f"shap_force_plot_class_{class_idx}_instance_0.html", force_plot)
print(f"✅ Saved interactive SHAP force plot for class {class_idx} at shap_force_plot_class_{class_idx}_instance_0.html")
print(f"✅ Saved SHAP summary & force plot for class {class_idx}")

print("\n✅ All SHAP visualizations generated and saved successfully.")
