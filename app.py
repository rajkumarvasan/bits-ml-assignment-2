import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

# Import your preprocessing function & model loader
from utils import preprocess_data, load_joblib_model

# App configuration
st.set_page_config(page_title="Machine Learning, Assignment - 2", layout="wide")
st.title("Mushroom Model Performance Dashboard")
st.write("Upload test dataset first, then choose a model from the dropdown to run the evaluation metrics.")

# Core configurations
MODEL_DIR = "model"
TARGET_COL = "class"  # Locked target column name

# Model name mapping
MODEL_NAME_MAP = {
    "Logistic Regression Classifier": "lr.joblib",
    "Decision Tree Classifier": "dt.joblib",
    "K-NN": "knn.joblib",
    "Naive Bayes": "nb.joblib",
    "Random Forest Classifier": "rf.joblib"
}

# Data Upload
st.subheader("Upload Test Dataset")
uploaded_file = st.file_uploader(
    "Drag and drop your 'mushrooms_test.csv' file here", 
    type=["csv"],
    help="Upload the raw categorical CSV test file containing the 'class' column."
)

# Main Dashboard Execution Flow triggers ONLY after file upload
if uploaded_file is not None:
    # Read and display data preview
    df = pd.read_csv(uploaded_file)
    
    # Validation check to ensure the required 'class' column exists
    if TARGET_COL not in df.columns:
        st.error(f"Critical Error: The uploaded dataset is missing the required '{TARGET_COL}' column.")
        st.stop()
        
    st.subheader(f"Test Data Preview ({uploaded_file.name})")
    st.dataframe(df.head(5))
    st.write("---")

    # Verify model directory exists before continuing
    if not os.path.exists(MODEL_DIR):
        st.error(f"Directory `{MODEL_DIR}/` not found. Please create it and add your `.joblib` files.")
        st.stop()

    # Keep only the models that actually exist in folder
    existing_custom_names = [
        display_name for display_name, filename in MODEL_NAME_MAP.items()
        if os.path.exists(os.path.join(MODEL_DIR, filename))
    ]

    if not existing_custom_names:
        st.warning(f"None of the mapped models were found in your `{MODEL_DIR}/` folder. Please verify filenames.")
        st.stop()

    # Model Selection
    st.subheader("Model Selection")
    
    # Add a blank indicator option at index 0
    model_options = ["--- Select a Model ---"] + sorted(existing_custom_names)
    selected_custom_name = st.selectbox("Choose a Pre-trained Model to Evaluate", model_options, index=0)

    # Execution halts here until the user chooses an actual custom model name
    if selected_custom_name == "--- Select a Model ---":
        st.info("Please select a model from the dropdown above to begin.")
    else:
        # Translate custom name back to its actual file path using the dictionary
        actual_filename = MODEL_NAME_MAP[selected_custom_name]
        full_model_path = os.path.join(MODEL_DIR, actual_filename)

        # Loads file to RAM only once per fresh choice
        if "current_model" not in st.session_state or st.session_state.get("active_model_path") != full_model_path:
            with st.spinner(f"Loading {selected_custom_name} into session memory..."):
                st.session_state["current_model"] = load_joblib_model(full_model_path)
                st.session_state["active_model_path"] = full_model_path
                st.toast(f"Loaded {selected_custom_name} to memory!")

        # Layout execution button
        st.write(" ") 
        run_evaluation = st.button("Run Evaluation Metrics", use_container_width=True, type="primary")

        # Predict and display metrics
        if run_evaluation:
            with st.spinner("Processing features and calculating scores..."):
                try:
                    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef

                    active_model = st.session_state["current_model"]

                    # Process features and target using your exact 2-value function format
                    X_processed, y_true = preprocess_data(df, TARGET_COL)

                    # Generate predictions
                    predictions = active_model.predict(X_processed)
                    # Extract probabilities for AUC calculation 
                    if hasattr(active_model, "predict_proba"):
                        y_prob = active_model.predict_proba(X_processed)[:, 1]
                    else:
                        y_prob = active_model.decision_function(X_processed)                    

                    # Map numerical outputs back to real world string variants for visualization 
                    unique_classes = np.unique(y_true)
                    class_labels_map = {0: "e (Edible)", 1: "p (Poisonous)"}
                    class_names = [class_labels_map.get(int(c), str(c)) for c in unique_classes]
                   
                    st.subheader("Evaluation Metrics")
                    col1, col2, col3, col4, col5, col6 = st.columns(6)
                    col1.metric("Accuracy", f"{accuracy_score(y_true, predictions):.2%}")
                    col2.metric("AUC Score", f"{roc_auc_score(y_true, y_prob):.2%}")
                    col3.metric("Precision", f"{precision_score(y_true, predictions):.2%}")
                    col4.metric("Recall", f"{recall_score(y_true, predictions):.2%}")
                    col5.metric("F1-Score", f"{f1_score(y_true, predictions):.2%}")
                    col6.metric("MCC Score", f"{matthews_corrcoef(y_true, predictions):.2%}")

                    st.write("---")
                    
                    # Render Grid Layout Columns for Charts 
                    vis_col1, vis_col2 = st.columns(2)

                    with vis_col1:
                        st.subheader("Confusion Matrix")
                        cm = confusion_matrix(y_true, predictions)
                        fig, ax = plt.subplots(figsize=(5, 4))
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                                    xticklabels=class_names, yticklabels=class_names, ax=ax)
                        plt.ylabel('Actual Label')
                        plt.xlabel('Predicted Label')
                        st.pyplot(fig)
                        plt.close(fig)

                    with vis_col2:
                        st.subheader("Classification Report")
                        report_dict = classification_report(y_true, predictions, target_names=class_names, output_dict=True)
                        report_df = pd.DataFrame(report_dict).transpose()
                        st.dataframe(report_df.style.format(precision=2))

                except Exception as e:
                    st.error(f"Evaluation Error: {e}")
else:
    st.info("Please upload CSV test dataset.")
