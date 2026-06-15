import pandas as pd
import numpy as np
import joblib, os, json
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

from preprocess import load_and_clean, get_X_y, fit_scaler, FEATURE_COLS

os.makedirs("assets", exist_ok=True)
os.makedirs("models", exist_ok=True)

def evaluate(model, X_test, y_test, name):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }
    cm = confusion_matrix(y_test, y_pred)
    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["No Fail", "Fail"], yticklabels=["No Fail", "Fail"])
    ax.set_title(f"{name} — Confusion Matrix")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"assets/cm_{name.lower().replace(' ','_')}.png", dpi=120)
    plt.close()
    print(f"\n{'='*40}\n{name}")
    for k, v in metrics.items():
        print(f"  {k:12s}: {v:.4f}")
    return metrics

def plot_feature_importance(model, feature_names, name):
    if not hasattr(model, "feature_importances_"):
        return
    imp = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    imp.plot(kind="barh", ax=ax, color="#4F8EF7")
    ax.set_title(f"{name} — Feature Importances")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig(f"assets/fi_{name.lower().replace(' ','_')}.png", dpi=120)
    plt.close()

def main():
    df = load_and_clean("data/vehicle_sensor_data.csv")
    X, y = get_X_y(df)
    X_scaled, scaler = fit_scaler(X)
    feature_names = X.columns.tolist()
    joblib.dump(feature_names, "models/feature_names.pkl")

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=15,
                                                 random_state=42, n_jobs=-1),
        "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(n_estimators=200, max_depth=6,
                                           learning_rate=0.1, use_label_encoder=False,
                                           eval_metric="logloss", random_state=42)

    results = {}
    trained = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        results[name] = evaluate(model, X_test, y_test, name)
        plot_feature_importance(model, feature_names, name)
        trained[name] = model

    # Select best by F1
    best_name = max(results, key=lambda k: results[k]["f1"])
    best_model = trained[best_name]
    print(f"\n✅ Best model: {best_name} (F1={results[best_name]['f1']:.4f})")

    joblib.dump(best_model, "models/best_model.pkl")
    joblib.dump(best_name, "models/best_model_name.pkl")

    # Save results
    with open("models/results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Bar chart comparison
    metrics_df = pd.DataFrame(results).T[["accuracy","precision","recall","f1","roc_auc"]]
    fig, ax = plt.subplots(figsize=(9, 5))
    metrics_df.plot(kind="bar", ax=ax, colormap="Set2", edgecolor="white")
    ax.set_title("Model Comparison", fontsize=14, fontweight="bold")
    ax.set_ylim(0.7, 1.02)
    ax.set_xticklabels(metrics_df.index, rotation=15)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("assets/model_comparison.png", dpi=130)
    plt.close()

    print("\nAll models saved. Assets generated.")
    return results

if __name__ == "__main__":
    main()
