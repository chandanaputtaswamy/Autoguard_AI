#!/usr/bin/env python3
"""
run_pipeline.py — One-shot setup: generate data → EDA → train models
Run this once before launching the Streamlit app.
"""
import subprocess, sys, os

def run(script, label):
    print(f"\n{'='*50}\n▶ {label}\n{'='*50}")
    result = subprocess.run([sys.executable, script], capture_output=False)
    if result.returncode != 0:
        print(f"❌ {label} failed (exit {result.returncode})")
        sys.exit(1)
    print(f"✅ {label} complete")

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, "src")

    run("src/generate_data.py", "Step 1/3 — Generate synthetic dataset")
    run("src/eda.py",           "Step 2/3 — Exploratory data analysis")
    run("src/train.py",         "Step 3/3 — Train & evaluate models")

    print("\n" + "="*50)
    print("🎉 Pipeline complete! Launch the dashboard with:")
    print("   streamlit run app.py")
    print("="*50)
