# import json
# from pathlib import Path
# import tempfile
# from app.utils.preprocess import extract_psg_features

# import shap

# import numpy as np
# import pandas as pd
# import streamlit as st

# from google.oauth2 import service_account
# from google.cloud import firestore
# import os
# import joblib

# import matplotlib.pyplot as plt

# # -----------------------------
# # Basic config
# # -----------------------------
# st.set_page_config(
#     page_title="SleepScope | Insomnia Analysis",
#     layout="wide",
# )

# st.markdown(
#     """
#     <style>
#     .big-title {
#         font-size: 2.4rem;
#         font-weight: 700;
#         margin-bottom: 0.2rem;
#     }
#     .subtitle {
#         font-size: 1rem;
#         color: #666666;
#         margin-bottom: 1.5rem;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True,
# )

# BASE_DIR = Path(__file__).resolve().parent
# MODELS_DIR = BASE_DIR / "models"

# # 🔧 CHANGE THIS to match your actual Firestore collection name
# FIRESTORE_COLLECTION = "sleepscope_sessions"  # TODO: update if different


# # -----------------------------
# # Helper: Load models safely
# # -----------------------------

# @st.cache_resource
# def load_psg_model():
#     try:
#         model_path = MODELS_DIR / "psg_model.pkl"
#         model = joblib.load(model_path)
#         return model
#     except Exception as e:
#         st.error(f"Error loading PSG model: {e}")
#         return None

# @st.cache_resource
# def load_subtype_pipeline():
#     """Load subtype model, scaler, feature list, and label map."""
#     try:
#         model = joblib.load(MODELS_DIR / "subtype_model.pkl")
#         scaler = joblib.load(MODELS_DIR / "subtype_scaler.pkl")

#         with open(MODELS_DIR / "subtype_features.json", "r") as f:
#             feature_names = json.load(f)

#         with open(MODELS_DIR / "subtype_label_map.json", "r") as f:
#             label_map = json.load(f)

#         # label_map is likely { "0": "Subtype A", "1": "Subtype B", ... }
#         # ensure keys are int
#         label_map = {int(k): v for k, v in label_map.items()}

#         return model, scaler, feature_names, label_map
#     except Exception as e:
#         st.error(f"Error loading subtype model or config: {e}")
#         return None, None, None, None


# @st.cache_resource
# # def get_firestore_client():
# #     """
# #     Returns a Firestore client.

# #     Assumes GOOGLE_APPLICATION_CREDENTIALS or equivalent is already
# #     configured in your Render environment (same as your FastAPI backend).
# #     """
# #     try:
# #         client = firestore.Client()
# #         return client
# #     except Exception as e:
# #         st.warning(
# #             "Could not connect to Firestore. "
# #             "Check credentials / environment variables."
# #         )
# #         st.info(str(e))
# #         return None

# def get_firestore_client():
#     """
#     Load Firestore using a service account JSON file.
#     Works both locally and on Render.
#     """
#     try:
#         cred_path = os.path.join("credentials", "serviceAccount.json")

#         if not os.path.exists(cred_path):
#             st.error(f"Service account file not found at: {cred_path}")
#             return None

#         credentials = service_account.Credentials.from_service_account_file(
#             cred_path
#         )

#         client = firestore.Client(
#             project=credentials.project_id,
#             credentials=credentials
#         )

#         return client

#     except Exception as e:
#         st.error(f"Could not connect to Firestore: {e}")
#         return None

# # -----------------------------
# # Helper: ISI severity based on total score
# # -----------------------------
# def get_isi_severity_label(isi_total: int) -> str:
#     """
#     Standard ISI interpretation:
#       0–7   : No clinically significant insomnia
#       8–14  : Subthreshold insomnia
#       15–21 : Moderate clinical insomnia
#       22–28 : Severe clinical insomnia
#     """
#     if isi_total <= 7:
#         return "No clinically significant insomnia"
#     elif isi_total <= 14:
#         return "Subthreshold insomnia"
#     elif isi_total <= 21:
#         return "Moderate clinical insomnia"
#     else:
#         return "Severe clinical insomnia"


# # -----------------------------
# # Helper: Predict subtype
# # -----------------------------
# def predict_subtype(feature_values: dict):
#     """
#     feature_values: dict { feature_name: float }
#     Uses feature_names from subtype_features.json to build ordered vector.
#     """
#     model, scaler, feature_names, label_map = load_subtype_pipeline()
#     if model is None or feature_names is None:
#         st.error("Subtype model not loaded. Please check model files.")
#         return None, None

#     # Build vector in the exact feature order
#     x = []
#     for name in feature_names:
#         val = feature_values.get(name, 0.0)  # default 0.0 if missing
#         x.append(float(val))

#     x = np.array(x).reshape(1, -1)

#     # Scale -> predict
#     try:
#         x_scaled = scaler.transform(x)
#     except Exception:
#         # If scaler is not available or fails, try without scaling
#         x_scaled = x

#     raw_pred = model.predict(x_scaled)[0]

#     # If model output is numeric class index, map to label
#     if isinstance(raw_pred, (int, np.integer)):
#         subtype_label = label_map.get(int(raw_pred), f"Class {raw_pred}")
#     else:
#         subtype_label = str(raw_pred)

#     return raw_pred, subtype_label


# def save_session_to_firestore(session_id, isi_total, phq9_total):
#     client = get_firestore_client()
#     if client is None:
#         st.error("Firestore connection failed. Scores could not be saved.")
#         return False

#     try:
#         client.collection(FIRESTORE_COLLECTION).document(session_id).set({
#             "session_id": session_id,
#             "isi_score": int(isi_total),
#             "phq9_score": int(phq9_total),
#             "timestamp": firestore.SERVER_TIMESTAMP,
#         })
#         return True

#     except Exception as e:
#         st.error(f"Error saving to Firestore: {e}")
#         return False



# # -----------------------------
# # Helper: Fetch ISI–PHQ9 correlation from Firestore
# # -----------------------------
# def fetch_isi_phq9_data():
#     client = get_firestore_client()
#     if client is None:
#         return pd.DataFrame()

#     try:
#         docs = client.collection(FIRESTORE_COLLECTION).stream()
#         rows = []
#         for d in docs:
#             data = d.to_dict()
#             # 🔧 CHANGE KEYS if your Firestore uses different field names
#             isi_val = data.get("isi_total") or data.get("isi_score")
#             phq_val = data.get("phq9_total") or data.get("phq9_score")
#             session_id = data.get("session_id", d.id)

#             if isi_val is not None and phq_val is not None:
#                 rows.append(
#                     {
#                         "session_id": session_id,
#                         "ISI": float(isi_val),
#                         "PHQ9": float(phq_val),
#                     }
#                 )

#         if not rows:
#             return pd.DataFrame()

#         return pd.DataFrame(rows)

#     except Exception as e:
#         st.error(f"Error reading from Firestore: {e}")
#         return pd.DataFrame()


# # -----------------------------
# # Layout: Main title
# # -----------------------------
# st.markdown(
#     """
#     <div class="big-title">SleepScope</div>
#     <div class="subtitle">
#         Insomnia Severity Prediction · Subtype Classification · Depression Correlation
#     </div>
#     """,
#     unsafe_allow_html=True,
# )

# tabs = st.tabs(
#     [
#         "Overview",
#         "User Dashboard",
#         "Clinician (PSG Upload)",
#         "Correlation Explorer",
#         "About / How it Works",
#     ]
# )

# # =====================================================
# #  TAB 1: OVERVIEW
# # =====================================================
# with tabs[0]:
#     col1, col2 = st.columns([2, 1])

#     with col1:
#         st.subheader("Project Summary")
#         st.write(
#             """
#             **SleepScope** is an explainable ML framework designed to:

#             - Estimate **insomnia severity**, based on ISI scores.
#             - Perform **insomnia subtype classification** using a trained ML model.
#             - Explore the **correlation between insomnia and depression**, via ISI and PHQ-9 scores stored in Firestore.
#             - Provide a **clinician-facing PSG upload section** for extending the analysis to polysomnography data.

#             This Streamlit app combines all components into a single, interactive dashboard
#             for both users and clinicians.
#             """
#         )

#     with col2:
#         st.markdown("### Demo Flow")
#         st.markdown(
#             """
#             1. Go to **User Dashboard**  
#                → Enter ISI & PHQ-9 totals  
#                → Get severity and subtype prediction.
               
#             2. Go to **Correlation Explorer**  
#                → View real-time ISI–PHQ9 correlation from Firestore.
               
#             3. Go to **Clinician (PSG)**  
#                → Upload PSG/EDF file (concept demo).  
#             """
#         )

# # =====================================================
# #  TAB 2: USER DASHBOARD (with full ISI + PHQ-9 questionnaires)
# # =====================================================
# with tabs[1]:
#     st.subheader("User Dashboard – ISI & PHQ-9 Questionnaires")

#     st.write(
#         """
#         Please answer the following **Insomnia Severity Index (ISI)**  
#         and **PHQ-9 Depression Assessment** questions.

#         Your total scores will be calculated automatically.
#         """
#     )

#     # -------------------------
#     # ISI QUESTIONS (0–4)
#     # -------------------------
#     st.markdown("## 💤 ISI – Insomnia Severity Index (0–28)")

#     isi_questions = [
#         "1. Difficulty falling asleep",
#         "2. Difficulty staying asleep",
#         "3. Problem waking up too early",
#         "4. Satisfaction with current sleep pattern",
#         "5. Noticeability of sleep problems to others",
#         "6. Worry/distress about sleep problems",
#         "7. Impact of sleep problems on daily functioning",
#     ]

#     isi_responses = []
#     isi_scale = ["0 = None", "1 = Mild", "2 = Moderate", "3 = Severe", "4 = Very Severe"]

#     for q in isi_questions:
#         val = st.select_slider(
#             q,
#             options=[0, 1, 2, 3, 4],
#             value=0,
#             help="0=None, 1=Mild, 2=Moderate, 3=Severe, 4=Very Severe"
#         )
#         isi_responses.append(val)

#     isi_total = sum(isi_responses)

#     st.info(f"**Total ISI Score: {isi_total}/28**")

#     # -------------------------
#     # PHQ-9 QUESTIONS (0–27)
#     # -------------------------
#     st.markdown("## 🧠 PHQ-9 – Depression Assessment (0–27)")

#     phq_questions = [
#         "1. Little interest or pleasure in doing things",
#         "2. Feeling down, depressed, or hopeless",
#         "3. Trouble falling or staying asleep, or sleeping too much",
#         "4. Feeling tired or having little energy",
#         "5. Poor appetite or overeating",
#         "6. Feeling bad about yourself or a failure",
#         "7. Trouble concentrating on things",
#         "8. Moving or speaking slowly OR being fidgety/restless",
#         "9. Thoughts of self-harm"
#     ]

#     phq_scale_labels = [
#         "0 = Not at all",
#         "1 = Several days",
#         "2 = More than half the days",
#         "3 = Nearly every day"
#     ]

#     phq_responses = []

#     for q in phq_questions:
#         val = st.select_slider(
#             q,
#             options=[0, 1, 2, 3],
#             value=0,
#             help="0=Not at all, 1=Several days, 2=More than half the days, 3=Nearly every day"
#         )
#         phq_responses.append(val)

#     phq9_total = sum(phq_responses)

#     st.info(f"**Total PHQ-9 Score: {phq9_total}/27**")

#     # ---------------------------------------------------------
#     # Subtype Feature Questions (clean, user-friendly inputs)
#     # ---------------------------------------------------------
#     st.markdown("## 🌙 Sleep & Lifestyle Questions for Subtype Classification")

#     st.write(
#         """
#         Please answer the following questions.  
#         These responses will be converted into normalized numerical features for the subtype model.
#         """
#     )

#     with st.container():
#         st.markdown("### 1️⃣ Sleep Duration")
#         sleep_duration_hours = st.number_input(
#             "How many hours of sleep do you usually get each night?",
#             min_value=0.0,
#             max_value=12.0,
#             value=6.0,
#             step=0.5
#         )

#         st.markdown("### 2️⃣ Sleep Quality")
#         sleep_quality = st.selectbox(
#             "How would you rate your overall sleep quality?",
#             ["Very Poor", "Poor", "Average", "Good", "Excellent"]
#         )

#         st.markdown("### 3️⃣ Daytime Sleepiness")
#         sleepiness = st.selectbox(
#             "How often do you feel sleepy or fatigued during the day?",
#             ["Never", "Rarely", "Sometimes", "Often", "Almost Always"]
#         )

#         st.markdown("### 4️⃣ General Stress Level")
#         stress_general = st.selectbox(
#             "How stressed do you feel in general?",
#             ["Not at all", "Mildly", "Moderately", "Highly", "Extremely"]
#         )

#         st.markdown("### 5️⃣ Anxiety Symptoms")
#         anxiety_score = st.selectbox(
#             "How often do you experience anxiety-related symptoms (e.g., worry, restlessness)?",
#             ["Never", "Rarely", "Sometimes", "Often", "Almost Always"]
#         )

#         st.markdown("### 6️⃣ BMI (Body Mass Index)")
#         bmi_value = st.number_input(
#             "Enter your BMI value:",
#             min_value=10.0,
#             max_value=45.0,
#             value=22.0,
#             step=0.1
#         )

#         # ---------------------------------------------------------
#         # Convert answers to normalized subtype feature values
#         # ---------------------------------------------------------

#         # Helper maps (0–1 scaling)
#         quality_map = {
#             "Very Poor": 0.0,
#             "Poor": 0.25,
#             "Average": 0.5,
#             "Good": 0.75,
#             "Excellent": 1.0
#         }

#         freq_map = {
#             "Never": 0.0,
#             "Rarely": 0.25,
#             "Sometimes": 0.5,
#             "Often": 0.75,
#             "Almost Always": 1.0
#         }

#         stress_map = {
#             "Not at all": 0.0,
#             "Mildly": 0.25,
#             "Moderately": 0.5,
#             "Highly": 0.75,
#             "Extremely": 1.0
#         }

#         # Normalize sleep duration to 0–1 (0–12 hours)
#         sleep_duration_norm = sleep_duration_hours / 12.0

#         # Convert all features
#         subtype_inputs = {
#             "sleep_duration": sleep_duration_norm,
#             "sleep_quality": quality_map[sleep_quality],
#             "sleepiness": freq_map[sleepiness],
#             "stress_general": stress_map[stress_general],
#             "anxiety_score": freq_map[anxiety_score],
#             "bmi": (bmi_value - 10) / 35  # normalize BMI 10–45 to 0–1
#         }



#     # -------------------------
#     # Submit Button
#     # -------------------------
#     if st.button("Run Analysis"):

#     # -------------------------------
#     # Save ISI & PHQ9 scores to Firestore
#     # -------------------------------
#         import uuid
#         session_id = str(uuid.uuid4())[:8]

#         saved = save_session_to_firestore(session_id, isi_total, phq9_total)

#         if saved:
#             st.success(f"Scores saved successfully (Session ID: {session_id})")
#         else:
#             st.error("Could not save scores to Firestore.")

#         # -------------------------------
#         # ISI Severity Prediction
#         # -------------------------------
#         severity_label = get_isi_severity_label(isi_total)

#         colA, colB = st.columns(2)

#         with colA:
#             st.markdown("### 💤 Insomnia Severity Result")
#             st.metric(
#                 label="Category",
#                 value=severity_label,
#                 delta=f"ISI = {isi_total}"
#             )

#         # -------------------------------
#         # Subtype Prediction (uses subtype_inputs)
#         # -------------------------------
#         with colB:
#             st.markdown("### 🔍 Insomnia Subtype (ML Model)")

#             _, _, subtype_feature_names, _ = load_subtype_pipeline()

#             # Filter only needed features
#             ordered_features = {
#                 f: subtype_inputs.get(f, 0.0) for f in subtype_feature_names
#                 if f in subtype_inputs
#             }

#             # Debug:
#             # st.write("Subtype model input:", ordered_features)

#             raw_pred, pretty_label = predict_subtype(ordered_features)

#             st.success(f"Predicted Subtype: **{pretty_label}**")
#             st.caption(f"Model Output: {raw_pred}")


# # # =====================================================
# # #  TAB 3: CLINICIAN (PSG UPLOAD WITH REAL PROCESSING)
# # # =====================================================
# # with tabs[2]:
# #     st.subheader("Clinician View – PSG / EDF Upload & Analysis")

# #     st.write(
# #         """
# #         Upload a **PSG (EDF) file** to extract features and run it through the
# #         trained PSG model. This demonstrates the back-end workflow used for
# #         polysomnography-based insomnia analysis.
# #         """
# #     )

# #     # uploaded_psg = st.file_uploader(
# #     #     "Upload PSG / EDF file",
# #     #     type=["edf", "EDF"],
# #     #     accept_multiple_files=False,
# #     # )

# #     uploaded_edf = st.file_uploader("Upload PSG / EDF file", type=["edf", "EDF"])
# #     uploaded_hyp = st.file_uploader("Upload Hypnogram File (.txt/.csv)", type=["txt", "csv"])

# #     if uploaded_edf and uploaded_hyp:
# #         st.success("Files uploaded successfully!")

# #         # Save EDF + Hypnogram to temp files
# #         with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as edf_tmp:
# #             edf_tmp.write(uploaded_edf.read())
# #             edf_path = edf_tmp.name

# #         with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as hyp_tmp:
# #             hyp_tmp.write(uploaded_hyp.read())
# #             hyp_path = hyp_tmp.name

# #         st.info("Extracting PSG features… please wait.")

# #         try:
# #             # Use the TRAINING PREPROCESSOR
# #             psg_features = extract_psg_features(edf_path, hyp_path)

# #             if psg_features is None:
# #                 st.error("PSG preprocessing returned no features.")
# #             else:
# #                 st.success("PSG features extracted successfully.")

# #                 # Convert dict → ordered vector
# #                 feature_names = list(psg_features.keys())
# #                 feature_vector = np.array(list(psg_features.values())).reshape(1, -1)

# #                 # Load model
# #                 psg_model = load_psg_model()
# #                 if psg_model is None:
# #                     st.error("PSG model could not be loaded.")
# #                 else:
# #                     prediction = psg_model.predict(feature_vector)[0]

# #                     st.subheader("PSG Model Prediction")
# #                     st.success(f"Predicted Output: **{prediction}**")

# #                     # SHAP
# #                     try:
# #                         st.markdown("### 🔍 SHAP Explanation")

# #                         explainer = shap.TreeExplainer(psg_model)
# #                         shap_values = explainer.shap_values(feature_vector)

# #                         st.write("#### Local Explanation")
# #                         fig = shap.force_plot(
# #                             explainer.expected_value,
# #                             shap_values[0],
# #                             feature_vector,
# #                             matplotlib=True
# #                         )
# #                         st.pyplot(fig)

# #                         st.write("#### Global Feature Importance")
# #                         fig2 = shap.summary_plot(shap_values, feature_vector, show=False)
# #                         st.pyplot(fig2)

# #                     except Exception as e:
# #                         st.warning(f"SHAP explanation could not be generated: {e}")

    

                    

# #                     # --------------------------
# #                     # Interpretation Section
# #                     # --------------------------
# #                     st.markdown("### 📌 Interpretation of Prediction")

# #                     # Example clinical interpretation logic
# #                     # Adjust thresholds later based on your model training
# #                     if prediction < 0:
# #                         st.warning(
# #                             "The predicted value is negative, which may indicate insufficient PSG data or a preprocessing issue. "
# #                             "Please verify the EDF file quality."
# #                         )

# #                     elif prediction < 0.2:
# #                         st.info(
# #                             """
# #                             **Low Risk / Mild Sleep Disturbance**

# #                             This score suggests *lower levels of physiological sleep disruption*.  
# #                             Features such as delta power, sleep continuity, and overall EEG stability  
# #                             remain within typical ranges.

# #                             Clinically, this may correspond to:
# #                             - Mild insomnia symptoms  
# #                             - Early-stage sleep disturbances  
# #                             - Psychophysiological insomnia  
# #                             """
# #                         )

# #                     elif prediction < 0.6:
# #                         st.warning(
# #                             """
# #                             **Moderate Risk / Noticeable Sleep Disruption**

# #                             The PSG features show **moderate deviation** from normal sleep architecture.  
# #                             This often includes irregularities in:
# #                             - Sleep stages  
# #                             - Micro-arousals  
# #                             - Reduced slow-wave (delta) activity  

# #                             Clinically, this may correspond to:
# #                             - Moderate insomnia  
# #                             - Stress-related sleep fragmentation  
# #                             """
# #                         )

# #                     else:
# #                         st.error(
# #                             """
# #                             **High Risk / Severe Sleep Disruption**

# #                             The model indicates **significant abnormalities** in EEG or sleep structure,  
# #                             such as:
# #                             - Markedly reduced restorative deep sleep  
# #                             - Increased arousal frequency  
# #                             - High instability in EEG spectral features  

# #                             Clinically, this may correspond to:
# #                             - Severe chronic insomnia  
# #                             - Underlying sleep disorders (e.g., sleep fragmentation disorder)  
# #                             """
# #                         )

# #                     # st.caption(
# #                     #     """
# #                     #     *Note: Interpretation thresholds are based on normalized model output.*
# #                     #     The exact thresholds depend on how the PSG model was trained.
# #                     #     """
# #                     # )


# #                     # st.caption(
# #                     #     """
# #                     #     This prediction is generated from:
# #                     #     - The EDF signal data you uploaded
# #                     #     - Your preprocessing pipeline in `app.utils.preprocess`
# #                     #     - The trained PSG model stored as `psg_model.pkl`
# #                     #     """
# #                     # )

# #         except Exception as e:
# #             st.error(f"Error processing PSG file: {e}")

# #     else:
# #         st.info("Please upload both EDF and Hypnogram files.")

# # =====================================================
# #  TAB 3: CLINICIAN (PSG + HYPNOGRAM UPLOAD)
# # =====================================================
# with tabs[2]:
#     st.subheader("Clinician View – PSG + Hypnogram Analysis")

#     st.write("Upload both the PSG file and its corresponding Hypnogram (.edf) file.")

#     col_psg, col_hyp = st.columns(2)

#     with col_psg:
#         uploaded_psg = st.file_uploader(
#             "Upload PSG / EDF File",
#             type=["edf", "EDF"],
#             key="psg_file"
#         )

#     with col_hyp:
#         uploaded_hyp = st.file_uploader(
#             "Upload Hypnogram / EDF File",
#             type=["edf", "EDF"],
#             key="hyp_file"
#         )

#     if uploaded_psg and uploaded_hyp:
#         st.success(f"Files uploaded:\n- PSG: {uploaded_psg.name}\n- Hypnogram: {uploaded_hyp.name}")

#         # Write to temp files
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp_psg:
#             tmp_psg.write(uploaded_psg.read())
#             psg_path = tmp_psg.name

#         with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp_hyp:
#             tmp_hyp.write(uploaded_hyp.read())
#             hyp_path = tmp_hyp.name

#         st.info("Extracting PSG features…")

#         try:
#             # Extract features (your original function)
#             psg_features = extract_psg_features(psg_path, hyp_path)

#             if psg_features is None:
#                 st.error("Feature extraction returned no values.")
#             else:
#                 st.success("PSG features extracted successfully!")
                

#                 # Convert dict → DataFrame row-like
#                 feature_order = [
#                    "TST_hours",
#                    "WASO_minutes",
#                    "SOL_minutes",
#                    "N1_minutes",
#                    "N2_minutes",
#                    "N3_minutes",
#                    "REM_minutes",
#                    "Sleep_Efficiency",
#                    "Total_Time_hours"
#                 ]                
                
#                 # st.write("Preprocessed PSG keys:", list(psg_features.keys()))
#                 # missing = [f for f in feature_order if f not in psg_features]
#                 # st.write("Missing features:", missing)
                
                

#                 feature_vector = np.array([psg_features[f] for f in feature_order]).reshape(1, -1)

#                 # Show extracted features
#                 # st.write("### Extracted PSG Features")
#                 # st.dataframe(pd.DataFrame([psg_features]))

#                 # Load model
#                 model_psg = load_psg_model()
#                 # st.write("Model expects features:", model_psg.get_booster().feature_names)


#                 prediction = model_psg.predict(feature_vector)[0]
#                 prediction = max(0.0, prediction)

#                 st.subheader("PSG Model Prediction")
#                 st.success(f"Predicted Output: **{prediction:.3f}**")

#                 # -----------------------
#                 # SHAP XAI
#                 # -----------------------
#                 st.markdown("### Explainable AI Interpretation")

#                 try:
#                     explainer = shap.TreeExplainer(model_psg)
#                     shap_output = explainer(feature_vector)

#                     explanation = shap.Explanation(
#                         values=shap_output.values[0],
#                         base_values=shap_output.base_values[0],
#                         data=feature_vector[0],
#                         feature_names=feature_order
#                     )

#                     # # Waterfall
#                     # st.write("####  Local Feature Contribution (Waterfall)")
#                     # fig1, ax1 = plt.subplots(figsize=(4,2))
#                     # shap.plots.waterfall(explanation, show=False)
#                     # st.pyplot(fig1, use_container_width=False)


#                     # # Bar plot
#                     # st.write("#### Global Feature Importance")
#                     # fig2, ax2 = plt.subplots(figsize=(4,2))
#                     # shap.summary_plot(shap_output.values, feature_vector,
#                     #                   feature_names=feature_order,
#                     #                   plot_type="bar", show=False)
#                     # st.pyplot(fig2, use_container_width=False)

#                     # Two column layout
#                     col1, col2 = st.columns(2)

#                     # ---------- Waterfall Plot ----------
#                     with col1:
#                         st.write("#### Local Feature Contribution (Waterfall)")
#                         fig1 = plt.figure(figsize=(5,4))  # smaller size to fit
#                         shap.plots.waterfall(explanation, show=False)
#                         st.pyplot(fig1, use_container_width=True)

#                     # ---------- Bar Plot ----------
#                     # with col2:
#                     #     st.write("#### Global Feature Importance (Bar)")
#                     #     fig2 = plt.figure(figsize=(5,4))
#                     #     shap.summary_plot(
#                     #         shap_output.values,
#                     #         feature_vector,
#                     #         feature_names=feature_order,
#                     #         plot_type="bar",
#                     #         show=False
#                     #     )
#                     #     st.pyplot(fig2, use_container_width=True)


#                 except Exception as e:
#                     st.error("SHAP explanation failed.")
#                     st.write(str(e))


#                     # --------------------------
#                     # Interpretation Section
#                     # --------------------------
#                     st.markdown("### Interpretation of Prediction")

#                     # Example clinical interpretation logic
#                     # Adjust thresholds later based on your model training
#                     if prediction < 0:
#                         st.warning(
#                             "The predicted value is negative, which may indicate insufficient PSG data or a preprocessing issue. "
#                             "Please verify the EDF file quality."
#                         )

#                     elif prediction < 0.2:
#                         st.info(
#                             """
#                             **Low Risk / Mild Sleep Disturbance**

#                             This score suggests *lower levels of physiological sleep disruption*.  
#                             Features such as delta power, sleep continuity, and overall EEG stability  
#                             remain within typical ranges.

#                             Clinically, this may correspond to:
#                             - Mild insomnia symptoms  
#                             - Early-stage sleep disturbances  
#                             - Psychophysiological insomnia  
#                             """
#                         )

#                     elif prediction < 0.6:
#                         st.warning(
#                             """
#                             **Moderate Risk / Noticeable Sleep Disruption**

#                             The PSG features show **moderate deviation** from normal sleep architecture.  
#                             This often includes irregularities in:
#                             - Sleep stages  
#                             - Micro-arousals  
#                             - Reduced slow-wave (delta) activity  

#                             Clinically, this may correspond to:
#                             - Moderate insomnia  
#                             - Stress-related sleep fragmentation  
#                             """
#                         )

#                     else:
#                         st.error(
#                             """
#                             **High Risk / Severe Sleep Disruption**

#                             The model indicates **significant abnormalities** in EEG or sleep structure,  
#                             such as:
#                             - Markedly reduced restorative deep sleep  
#                             - Increased arousal frequency  
#                             - High instability in EEG spectral features  

#                             Clinically, this may correspond to:
#                             - Severe chronic insomnia  
#                             - Underlying sleep disorders (e.g., sleep fragmentation disorder)  
#                             """
#                         )

#                     # st.caption(
#                     #     """
#                     #     *Note: Interpretation thresholds are based on normalized model output.*
#                     #     The exact thresholds depend on how the PSG model was trained.
#                     #     """
#                     # )


#                     # st.caption(
#                     #     """
#                     #     This prediction is generated from:
#                     #     - The EDF signal data you uploaded
#                     #     - Your preprocessing pipeline in `app.utils.preprocess`
#                     #     - The trained PSG model stored as `psg_model.pkl`
#                     #     """
#                     # )

#         except Exception as e:
#             st.error(f"Error processing PSG file: {e}")



# # =====================================================
# #  TAB 4: CORRELATION EXPLORER
# # =====================================================
# with tabs[3]:
#     st.subheader("Correlation Explorer – ISI vs PHQ-9")

#     st.write(
#         """
#         This section computes and visualizes the **correlation between insomnia severity**
#         and **depression symptoms** using ISI and PHQ-9 scores stored in Firestore.
#         """
#     )

#     df_corr = fetch_isi_phq9_data()

#     if df_corr.empty:
#         st.warning(
#             "No data found in Firestore or unable to connect. "
#             "Ensure the collection name and credentials are correct."
#         )
#     else:
#         # st.markdown("#### Sample Data")
#         # st.dataframe(df_corr.head())

#         corr_val = df_corr[["ISI", "PHQ9"]].corr().iloc[0, 1]
#         st.metric(
#             label="Pearson Correlation (ISI vs PHQ-9)",
#             value=f"{corr_val:.3f}",
#         )

#         st.markdown("#### Scatter Plot")
#         st.write(
#             "Each point represents a **session** with both ISI and PHQ-9 scores."
#         )
#         st.scatter_chart(df_corr, x="ISI", y="PHQ9")

#         st.caption(
#             """
#             A higher positive correlation suggests that higher insomnia severity
#             is associated with higher depression scores in the observed population.
#             """
#         )


# # =====================================================
# #  TAB 5: ABOUT / HOW IT WORKS
# # =====================================================
# with tabs[4]:
#     st.subheader("About SleepScope & Technical Workflow")

#     st.markdown(
#         """
#         ### 1. Data Sources

#         - **Questionnaire data**:  
#           - Insomnia Severity Index (ISI) – severity of insomnia  
#           - PHQ-9 – depression symptoms  
#         - **Optional PSG data** (EDF):
#           - EEG/EOG/EMG channels extracted as features for advanced modelling.

#         ### 2. ML Components

#         1. **ISI-based Severity Estimation (Rule-based)**  
#            - ISI total score is categorized into severity levels:
#              - 0–7: No clinically significant insomnia  
#              - 8–14: Subthreshold insomnia  
#              - 15–21: Moderate clinical insomnia  
#              - 22–28: Severe clinical insomnia  

#         2. **Subtype Classification (ML Model)**  
#            - Uses `subtype_model.pkl`, `subtype_scaler.pkl`, and
#              `subtype_features.json`.  
#            - Features are collected from the user and scaled before prediction.  
#            - Output label is mapped via `subtype_label_map.json`.

#         3. **Depression Correlation (Explainable Insight)**  
#            - ISI and PHQ-9 scores are stored in Firestore along with a session ID.  
#            - Correlation between ISI and PHQ-9 is computed and visualized to
#              study how insomnia might co-occur with depression.

#         ### 3. Architecture

#         - **Backend** (already deployed on Render):  
#           - Handles model training / storage, scoring logic, and Firestore integration.
#         - **This Streamlit app** (same repo):
#           - Acts as a lightweight, Python-based frontend.
#           - Directly loads `.pkl` models and Firestore data.
#           - Provides separate views for **users** and **clinicians**.

#         ### 4. Why Streamlit?

#         - Rapid prototype for demo under strict time constraints.
#         - Eliminates complex JS–backend integration issues.
#         - Still demonstrates:
#           - End-to-end ML workflow  
#           - Data pipeline  
#           - Real-time analytics & explainability  
#         """
#     )


import json
from pathlib import Path
import tempfile
from app.utils.preprocess import extract_psg_features
import logging

import shap
import matplotlib.pyplot as plt
shap.initjs()

import numpy as np
import pandas as pd
import streamlit as st

from google.oauth2 import service_account
from google.cloud import firestore
import os
import joblib


# -----------------------------
# Basic config
# -----------------------------
st.set_page_config(
    page_title="SleepScope | Insomnia Analysis",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Global Styling */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main container */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
    }
    
    /* Title Styling */
    .big-title {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
        letter-spacing: -1px;
        text-align: center;
    }
    
    .subtitle {
        font-size: 1.2rem;
        color: #6c757d;
        margin-bottom: 2rem;
        text-align: center;
        font-weight: 400;
    }
    
    /* Card Styling */
    .stContainer {
        background: white;
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .stContainer:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 50px rgba(0,0,0,0.15);
    }
    
    /* Button Styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        width: 100%;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Metric Styling */
    .stMetric {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.07);
    }
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f8f9fa;
        border-radius: 12px;
        padding: 0.5rem;
        justify-content: center;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #495057;
        font-weight: 600;
        padding: 0.75rem 1.5rem;
        transition: all 0.3s ease;
        font-size: 1.1rem;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Remove the orange underline indicator */
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab-border"] {
        background-color: transparent;
    }
    
    /* Info/Success/Warning boxes */
    .stAlert {
        border-radius: 12px;
        border-left: 4px solid;
        padding: 1rem 1.5rem;
    }
    
    /* Select slider styling */
    .stSlider {
        padding: 1rem 0;
    }
    
    /* Subheader styling */
    h2, h3 {
        color: #2d3748;
        font-weight: 700;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    
    /* Input field styling */
    .stNumberInput>div>div>input,
    .stSelectbox>div>div>select {
        border-radius: 8px;
        border: 2px solid #e2e8f0;
        padding: 0.5rem;
        transition: border-color 0.3s ease;
    }
    
    .stNumberInput>div>div>input:focus,
    .stSelectbox>div>div>select:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* File uploader styling */
    .stFileUploader {
        border: 2px dashed #cbd5e0;
        border-radius: 12px;
        padding: 2rem;
        transition: border-color 0.3s ease;
    }
    
    .stFileUploader:hover {
        border-color: #667eea;
    }
    
    /* Markdown content styling */
    .markdown-text-container {
        color: #4a5568;
        line-height: 1.7;
    }
    
    /* Section divider */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
        margin: 2rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

# 🔧 CHANGE THIS to match your actual Firestore collection name
FIRESTORE_COLLECTION = "sleepscope_sessions"  # TODO: update if different


# -----------------------------
# Helper: Load models safely
# -----------------------------

@st.cache_resource
def load_psg_model():
    try:
        model_path = MODELS_DIR / "psg_model.pkl"
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading PSG model: {e}")
        return None

@st.cache_resource
@st.cache_resource
def load_subtype_pipeline():
    """Load subtype model, scaler, feature list, and label map."""
    try:
        model = joblib.load(MODELS_DIR / "subtype_model.pkl")
        scaler = joblib.load(MODELS_DIR / "subtype_scaler.pkl")

        with open(MODELS_DIR / "subtype_features.json", "r") as f:
            feature_names = json.load(f)

        with open(MODELS_DIR / "subtype_label_map.json", "r") as f:
            label_map = json.load(f)

        # label_map is likely { "0": "Subtype A", "1": "Subtype B", ... }
        # ensure keys are int
        label_map = {int(k): v for k, v in label_map.items()}

        return model, scaler, feature_names, label_map
    except Exception as e:
        st.error(f"Error loading subtype model or config: {e}")
        return None, None, None, None


@st.cache_resource
# def get_firestore_client():
#     """
#     Returns a Firestore client.

#     Assumes GOOGLE_APPLICATION_CREDENTIALS or equivalent is already
#     configured in your Render environment (same as your FastAPI backend).
#     """
#     try:
#         client = firestore.Client()
#         return client
#     except Exception as e:
#         st.warning(
#             "Could not connect to Firestore. "
#             "Check credentials / environment variables."
#         )
#         st.info(str(e))
#         return None

def get_firestore_client():
    """
    Load Firestore using a service account JSON file.
    Works both locally and on Render.
    """
    try:
        cred_path = os.path.join("credentials", "serviceAccount.json")

        if not os.path.exists(cred_path):
            st.error(f"Service account file not found at: {cred_path}")
            return None

        credentials = service_account.Credentials.from_service_account_file(
            cred_path
        )

        client = firestore.Client(
            project=credentials.project_id,
            credentials=credentials
        )

        return client

    except Exception as e:
        st.error(f"Could not connect to Firestore: {e}")
        return None

# -----------------------------
# Helper: ISI severity based on total score
# -----------------------------
def get_isi_severity_label(isi_total: int) -> str:
    """
    Standard ISI interpretation:
      0–7   : No clinically significant insomnia
      8–14  : Subthreshold insomnia
      15–21 : Moderate clinical insomnia
      22–28 : Severe clinical insomnia
    """
    if isi_total <= 7:
        return "No clinically significant insomnia"
    elif isi_total <= 14:
        return "Subthreshold insomnia"
    elif isi_total <= 21:
        return "Moderate clinical insomnia"
    else:
        return "Severe clinical insomnia"


# -----------------------------
# Helper: Predict subtype
# -----------------------------
def predict_subtype(feature_values: dict):
    """
    feature_values: dict { feature_name: float }
    Uses feature_names from subtype_features.json to build ordered vector.
    """
    model, scaler, feature_names, label_map = load_subtype_pipeline()
    if model is None or feature_names is None:
        st.error("Subtype model not loaded. Please check model files.")
        return None, None

    # Build vector in the exact feature order
    x = []
    for name in feature_names:
        val = feature_values.get(name, 0.0)  # default 0.0 if missing
        x.append(float(val))

    x = np.array(x).reshape(1, -1)

    # Scale -> predict
    try:
        x_scaled = scaler.transform(x)
    except Exception:
        # If scaler is not available or fails, try without scaling
        x_scaled = x

    raw_pred = model.predict(x_scaled)[0]

    # If model output is numeric class index, map to label
    if isinstance(raw_pred, (int, np.integer)):
        subtype_label = label_map.get(int(raw_pred), f"Class {raw_pred}")
    else:
        subtype_label = str(raw_pred)

    return raw_pred, subtype_label


def save_session_to_firestore(session_id, isi_total, phq9_total):
    client = get_firestore_client()
    if client is None:
        st.error("Firestore connection failed. Scores could not be saved.")
        return False

    try:
        client.collection(FIRESTORE_COLLECTION).document(session_id).set({
            "session_id": session_id,
            "isi_score": int(isi_total),
            "phq9_score": int(phq9_total),
            "timestamp": firestore.SERVER_TIMESTAMP,
        })
        return True

    except Exception as e:
        st.error(f"Error saving to Firestore: {e}")
        return False



# -----------------------------
# Helper: Fetch ISI–PHQ9 correlation from Firestore
# -----------------------------
def fetch_isi_phq9_data():
    client = get_firestore_client()
    if client is None:
        return pd.DataFrame()

    try:
        docs = client.collection(FIRESTORE_COLLECTION).stream()
        rows = []
        for d in docs:
            data = d.to_dict()
            # CHANGE KEYS if your Firestore uses different field names
            isi_val = data.get("isi_total") or data.get("isi_score")
            phq_val = data.get("phq9_total") or data.get("phq9_score")
            session_id = data.get("session_id", d.id)

            if isi_val is not None and phq_val is not None:
                rows.append(
                    {
                        "session_id": session_id,
                        "ISI": float(isi_val),
                        "PHQ9": float(phq_val),
                    }
                )

        if not rows:
            return pd.DataFrame()

        return pd.DataFrame(rows)

    except Exception as e:
        st.error(f"Error reading from Firestore: {e}")
        return pd.DataFrame()


# -----------------------------
# Layout: Main title with icon
# -----------------------------
st.markdown(
    """
    <div class="big-title">SleepScope</div>
    <div class="subtitle">
        Insomnia Severity Prediction · Subtype Classification · Depression Correlation
    </div>
    """,
    unsafe_allow_html=True,
)

tabs = st.tabs(
    [
        "Overview",
        "User Dashboard",
        "Clinician (PSG Upload)",
        "Correlation Explorer",
    ]
)

# =====================================================
#  TAB 1: OVERVIEW
# =====================================================
with tabs[0]:
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Hero section
    st.markdown("""
        <div style='text-align: center; padding: 2rem 0;'>
            <h1 style='font-size: 2.5rem; color: #2d3748; margin-bottom: 1rem;'>
                Welcome to SleepScope
            </h1>
            <p style='font-size: 1.2rem; color: #718096; max-width: 800px; margin: 0 auto;'>
                An advanced explainable ML framework for comprehensive insomnia analysis
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Feature cards in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 2rem; border-radius: 16px; color: white; text-align: center;
                        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);'>
                <div style='width: 60px; height: 60px; background: rgba(255,255,255,0.2); 
                            border-radius: 50%; margin: 0 auto 1rem; display: flex; 
                            align-items: center; justify-content: center; font-size: 1.5rem;'>
                    <strong>ISI</strong>
                </div>
                <h3 style='color: white; margin: 1rem 0 0.5rem 0;'>Severity Estimation</h3>
                <p style='color: rgba(255,255,255,0.9); font-size: 0.95rem;'>
                    ISI-based insomnia severity classification
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 2rem; border-radius: 16px; color: white; text-align: center;
                        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);'>
                <div style='width: 60px; height: 60px; background: rgba(255,255,255,0.2); 
                            border-radius: 50%; margin: 0 auto 1rem; display: flex; 
                            align-items: center; justify-content: center; font-size: 1.5rem;'>
                    <strong>ML</strong>
                </div>
                <h3 style='color: white; margin: 1rem 0 0.5rem 0;'>Subtype Analysis</h3>
                <p style='color: rgba(255,255,255,0.9); font-size: 0.95rem;'>
                    ML-powered insomnia subtype classification
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 2rem; border-radius: 16px; color: white; text-align: center;
                        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);'>
                <div style='width: 60px; height: 60px; background: rgba(255,255,255,0.2); 
                            border-radius: 50%; margin: 0 auto 1rem; display: flex; 
                            align-items: center; justify-content: center; font-size: 1.5rem;'>
                    <strong>r</strong>
                </div>
                <h3 style='color: white; margin: 1rem 0 0.5rem 0;'>Correlation Insights</h3>
                <p style='color: rgba(255,255,255,0.9); font-size: 0.95rem;'>
                    ISI-PHQ9 depression correlation analysis
                </p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    # Main content
    with st.container():
        st.markdown("### What is SleepScope?")
        st.write(
            """
            **SleepScope** is an advanced explainable machine learning framework that provides:

            - **Comprehensive Assessment**: Multi-dimensional evaluation combining ISI scores, 
              PHQ-9 depression metrics, and optional polysomnography data
            - **ML-Driven Insights**: State-of-the-art models for accurate subtype classification
            - **Clinical Utility**: Designed for both patient self-assessment and clinical decision support
            - **Explainable AI**: Transparent predictions with SHAP-based interpretability
            - **Real-time Analytics**: Live correlation analysis using cloud-stored patient data

            This integrated dashboard serves both end-users and healthcare professionals with 
            a unified interface for insomnia analysis and monitoring.
            """
        )

# =====================================================
#  TAB 2: USER DASHBOARD (with full ISI + PHQ-9 questionnaires)
# =====================================================
with tabs[1]:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
        <div style='text-align: center; margin-bottom: 2rem;'>
            <h1 style='color: #2d3748; font-size: 2.2rem;'>User Assessment Dashboard</h1>
            <p style='color: #718096; font-size: 1.1rem;'>Complete the ISI and PHQ-9 questionnaires for personalized analysis</p>
        </div>
    """, unsafe_allow_html=True)

    # -------------------------
    # ISI QUESTIONS (0–4)
    # -------------------------
    st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 1.5rem; border-radius: 12px; margin-bottom: 2rem;'>
            <h2 style='color: white; margin: 0; font-size: 1.8rem;'>ISI – Insomnia Severity Index</h2>
            <p style='color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0;'>Score Range: 0–28</p>
        </div>
    """, unsafe_allow_html=True)

    isi_questions = [
        "1. Difficulty falling asleep",
        "2. Difficulty staying asleep",
        "3. Problem waking up too early",
        "4. Satisfaction with current sleep pattern",
        "5. Noticeability of sleep problems to others",
        "6. Worry/distress about sleep problems",
        "7. Impact of sleep problems on daily functioning",
    ]

    isi_responses = []
    isi_scale = ["0 = None", "1 = Mild", "2 = Moderate", "3 = Severe", "4 = Very Severe"]

    for q in isi_questions:
        val = st.select_slider(
            q,
            options=[0, 1, 2, 3, 4],
            value=0,
            help="0=None, 1=Mild, 2=Moderate, 3=Severe, 4=Very Severe"
        )
        isi_responses.append(val)

    isi_total = sum(isi_responses)

    st.markdown(f"""
        <div style='background: #e6f3ff; padding: 1rem; border-radius: 8px; 
                    border-left: 4px solid #667eea; margin: 1rem 0;'>
            <strong style='color: #2d3748; font-size: 1.2rem;'>Total ISI Score: {isi_total}/28</strong>
        </div>
    """, unsafe_allow_html=True)

    # -------------------------
    # PHQ-9 QUESTIONS (0–27)
    # -------------------------
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 1.5rem; border-radius: 12px; margin-bottom: 2rem;'>
            <h2 style='color: white; margin: 0; font-size: 1.8rem;'>PHQ-9 – Depression Assessment</h2>
            <p style='color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0;'>Score Range: 0–27</p>
        </div>
    """, unsafe_allow_html=True)

    phq_questions = [
        "1. Little interest or pleasure in doing things",
        "2. Feeling down, depressed, or hopeless",
        "3. Trouble falling or staying asleep, or sleeping too much",
        "4. Feeling tired or having little energy",
        "5. Poor appetite or overeating",
        "6. Feeling bad about yourself or a failure",
        "7. Trouble concentrating on things",
        "8. Moving or speaking slowly OR being fidgety/restless",
        "9. Thoughts of self-harm"
    ]

    phq_scale_labels = [
        "0 = Not at all",
        "1 = Several days",
        "2 = More than half the days",
        "3 = Nearly every day"
    ]

    phq_responses = []

    for q in phq_questions:
        val = st.select_slider(
            q,
            options=[0, 1, 2, 3],
            value=0,
            help="0=Not at all, 1=Several days, 2=More than half the days, 3=Nearly every day"
        )
        phq_responses.append(val)

    phq9_total = sum(phq_responses)

    st.markdown(f"""
        <div style='background: #e6f3ff; padding: 1rem; border-radius: 8px; 
                    border-left: 4px solid #667eea; margin: 1rem 0;'>
            <strong style='color: #2d3748; font-size: 1.2rem;'>Total PHQ-9 Score: {phq9_total}/27</strong>
        </div>
    """, unsafe_allow_html=True)

    # ---------------------------------------------------------
    # Subtype Feature Questions (clean, user-friendly inputs)
    # ---------------------------------------------------------
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 1.5rem; border-radius: 12px; margin-bottom: 2rem;'>
            <h2 style='color: white; margin: 0; font-size: 1.8rem;'>Sleep & Lifestyle Assessment</h2>
            <p style='color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0;'>For subtype classification</p>
        </div>
    """, unsafe_allow_html=True)

    with st.container():
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.markdown("#### Sleep Duration")
            sleep_duration_hours = st.number_input(
                "How many hours of sleep do you usually get each night?",
                min_value=0.0,
                max_value=12.0,
                value=6.0,
                step=0.5
            )

            st.markdown("#### Sleep Quality")
            sleep_quality = st.selectbox(
                "How would you rate your overall sleep quality?",
                ["Very Poor", "Poor", "Average", "Good", "Excellent"]
            )

            st.markdown("#### Daytime Sleepiness")
            sleepiness = st.selectbox(
                "How often do you feel sleepy or fatigued during the day?",
                ["Never", "Rarely", "Sometimes", "Often", "Almost Always"]
            )
        
        with col_right:
            st.markdown("#### General Stress Level")
            stress_general = st.selectbox(
                "How stressed do you feel in general?",
                ["Not at all", "Mildly", "Moderately", "Highly", "Extremely"]
            )

            st.markdown("#### Anxiety Symptoms")
            anxiety_score = st.selectbox(
                "How often do you experience anxiety-related symptoms (e.g., worry, restlessness)?",
                ["Never", "Rarely", "Sometimes", "Often", "Almost Always"]
            )

            st.markdown("#### BMI (Body Mass Index)")
            bmi_value = st.number_input(
                "Enter your BMI value:",
                min_value=10.0,
                max_value=45.0,
                value=22.0,
                step=0.1
            )

        # ---------------------------------------------------------
        # Convert answers to normalized subtype feature values
        # ---------------------------------------------------------

        # Helper maps (0–1 scaling)
        quality_map = {
            "Very Poor": 0.0,
            "Poor": 0.25,
            "Average": 0.5,
            "Good": 0.75,
            "Excellent": 1.0
        }

        freq_map = {
            "Never": 0.0,
            "Rarely": 0.25,
            "Sometimes": 0.5,
            "Often": 0.75,
            "Almost Always": 1.0
        }

        stress_map = {
            "Not at all": 0.0,
            "Mildly": 0.25,
            "Moderately": 0.5,
            "Highly": 0.75,
            "Extremely": 1.0
        }

        # Normalize sleep duration to 0–1 (0–12 hours)
        sleep_duration_norm = sleep_duration_hours / 12.0

        # Convert all features
        subtype_inputs = {
            "sleep_duration": sleep_duration_norm,
            "sleep_quality": quality_map[sleep_quality],
            "sleepiness": freq_map[sleepiness],
            "stress_general": stress_map[stress_general],
            "anxiety_score": freq_map[anxiety_score],
            "bmi": (bmi_value - 10) / 35  # normalize BMI 10–45 to 0–1
        }



    # -------------------------
    # Submit Button
    # -------------------------
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Run Complete Analysis", use_container_width=True):

    # -------------------------------
    # Save ISI & PHQ9 scores to Firestore
    # -------------------------------
        import uuid
        session_id = str(uuid.uuid4())[:8]

        saved = save_session_to_firestore(session_id, isi_total, phq9_total)

        if saved:
            logging.info(f"Scores saved successfully (Session ID: {session_id})")
        else:
            logging.error("Could not save scores to Firestore.")
        
        st.markdown("<br>", unsafe_allow_html=True)

        # -------------------------------
        # ISI Severity Prediction
        # -------------------------------
        severity_label = get_isi_severity_label(isi_total)

        colA, colB = st.columns(2)

        with colA:
            st.markdown("""
                <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                            padding: 2rem; border-radius: 12px; text-align: center; color: white;
                            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);'>
                    <h3 style='color: white; margin: 0 0 1rem 0;'>Insomnia Severity</h3>
                    <h2 style='color: white; font-size: 1.5rem; margin: 0;'>{}</h2>
                    <p style='color: rgba(255,255,255,0.9); margin: 1rem 0 0 0;'>ISI Score: {}</p>
                </div>
            """.format(severity_label, isi_total), unsafe_allow_html=True)

        # -------------------------------
        # Subtype Prediction (uses subtype_inputs)
        # -------------------------------
        with colB:
            _, _, subtype_feature_names, _ = load_subtype_pipeline()

            # Filter only needed features
            ordered_features = {
                f: subtype_inputs.get(f, 0.0) for f in subtype_feature_names
                if f in subtype_inputs
            }

            # Debug:
            # st.write("Subtype model input:", ordered_features)

            raw_pred, pretty_label = predict_subtype(ordered_features)

            st.markdown(f"""
                <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                            padding: 2rem; border-radius: 12px; text-align: center; color: white;
                            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);'>
                    <h3 style='color: white; margin: 0 0 1rem 0;'>Insomnia Subtype</h3>
                    <h2 style='color: white; font-size: 1.5rem; margin: 0;'>{pretty_label}</h2>
                    <p style='color: rgba(255,255,255,0.9); margin: 1rem 0 0 0;'>Model Output: {raw_pred}</p>
                </div>
            """, unsafe_allow_html=True)


# =====================================================
#  TAB 3: CLINICIAN (PSG + HYPNOGRAM UPLOAD)
# =====================================================
with tabs[2]:
    st.subheader("Clinician View – PSG + Hypnogram Analysis")

    st.write("Upload both the PSG file and its corresponding Hypnogram (.edf) file.")

    col_psg, col_hyp = st.columns(2)

    with col_psg:
        uploaded_psg = st.file_uploader(
            "Upload PSG / EDF File",
            type=["edf", "EDF"],
            key="psg_file"
        )

    with col_hyp:
        uploaded_hyp = st.file_uploader(
            "Upload Hypnogram / EDF File",
            type=["edf", "EDF"],
            key="hyp_file"
        )

    if uploaded_psg and uploaded_hyp:
        st.success(f"Files uploaded:\n- PSG: {uploaded_psg.name}\n- Hypnogram: {uploaded_hyp.name}")

        # Write to temp files
        with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp_psg:
            tmp_psg.write(uploaded_psg.read())
            psg_path = tmp_psg.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp_hyp:
            tmp_hyp.write(uploaded_hyp.read())
            hyp_path = tmp_hyp.name

        st.info("Extracting PSG features…")

        try:
            # Extract features (your original function)
            psg_features = extract_psg_features(psg_path, hyp_path)

            if psg_features is None:
                st.error("Feature extraction returned no values.")
            else:
                st.success("PSG features extracted successfully!")
                

                # Convert dict → DataFrame row-like
                feature_order = [
                    "TST_hours",
                    "WASO_rate",
                    "SOL_rate",
                    "N1_ratio",
                    "N2_ratio",
                    "N3_ratio",
                    "REM_ratio",
                    "Sleep_Efficiency",
                    "Fragmentation",
                    "Total_Time_hours",
                ]             
                
                # st.write("Preprocessed PSG keys:", list(psg_features.keys()))
                # missing = [f for f in feature_order if f not in psg_features]
                # st.write("Missing features:", missing)
                
                

                feature_vector = np.array([psg_features[f] for f in feature_order]).reshape(1, -1)

                # Show extracted features
                # st.write("### Extracted PSG Features")
                # st.dataframe(pd.DataFrame([psg_features]))

                # Load model
                model_psg = load_psg_model()
                # st.write("Model expects features:", model_psg.get_booster().feature_names)


                prediction = model_psg.predict(feature_vector)[0]
                prediction = max(0.0, prediction)

                st.subheader("PSG Model Prediction")
                st.success(f"Predicted Output: **{prediction:.3f}**")

                # -----------------------
                # SHAP XAI
                # -----------------------
                st.markdown("### Explainable AI Interpretation")

                try:
                    explainer = shap.TreeExplainer(model_psg)
                    shap_output = explainer(feature_vector)

                    explanation = shap.Explanation(
                        values=shap_output.values[0],
                        base_values=shap_output.base_values[0],
                        data=feature_vector[0],
                        feature_names=feature_order
                    )

                    # # Waterfall
                    # st.write("####  Local Feature Contribution (Waterfall)")
                    # fig1, ax1 = plt.subplots(figsize=(4,2))
                    # shap.plots.waterfall(explanation, show=False)
                    # st.pyplot(fig1, use_container_width=False)


                    # # Bar plot
                    # st.write("#### Global Feature Importance")
                    # fig2, ax2 = plt.subplots(figsize=(4,2))
                    # shap.summary_plot(shap_output.values, feature_vector,
                    #                   feature_names=feature_order,
                    #                   plot_type="bar", show=False)
                    # st.pyplot(fig2, use_container_width=False)

                    # Two column layout
                    col1, col2 = st.columns(2)

                    # ---------- Waterfall Plot ----------
                    with col1:
                        st.write("#### Local Feature Contribution (Waterfall)")
                        fig1 = plt.figure(figsize=(5,4))  # smaller size to fit
                        shap.plots.waterfall(explanation, show=False)
                        st.pyplot(fig1, use_container_width=True)

                    # ---------- Bar Plot ----------
                    with col2:
                        st.write("#### Global Feature Importance (Bar)")
                        fig2 = plt.figure(figsize=(5,4))
                        shap.summary_plot(
                            shap_output.values,
                            feature_vector,
                            feature_names=feature_order,
                            plot_type="bar",
                            show=False
                        )
                        st.pyplot(fig2, use_container_width=True)


                except Exception as e:
                    st.error("SHAP explanation failed.")
                    st.write(str(e))


                    # --------------------------
                    # Interpretation Section
                    # --------------------------
                    st.markdown("### Interpretation of Prediction")

                    # Example clinical interpretation logic
                    # Adjust thresholds later based on your model training
                    if prediction < 0:
                        st.warning(
                            "The predicted value is negative, which may indicate insufficient PSG data or a preprocessing issue. "
                            "Please verify the EDF file quality."
                        )

                    elif prediction < 0.2:
                        st.info(
                            """
                            **Low Risk / Mild Sleep Disturbance**

                            This score suggests *lower levels of physiological sleep disruption*.  
                            Features such as delta power, sleep continuity, and overall EEG stability  
                            remain within typical ranges.

                            Clinically, this may correspond to:
                            - Mild insomnia symptoms  
                            - Early-stage sleep disturbances  
                            - Psychophysiological insomnia  
                            """
                        )

                    elif prediction < 0.6:
                        st.warning(
                            """
                            **Moderate Risk / Noticeable Sleep Disruption**

                            The PSG features show **moderate deviation** from normal sleep architecture.  
                            This often includes irregularities in:
                            - Sleep stages  
                            - Micro-arousals  
                            - Reduced slow-wave (delta) activity  

                            Clinically, this may correspond to:
                            - Moderate insomnia  
                            - Stress-related sleep fragmentation  
                            """
                        )

                    else:
                        st.error(
                            """
                            **High Risk / Severe Sleep Disruption**

                            The model indicates **significant abnormalities** in EEG or sleep structure,  
                            such as:
                            - Markedly reduced restorative deep sleep  
                            - Increased arousal frequency  
                            - High instability in EEG spectral features  

                            Clinically, this may correspond to:
                            - Severe chronic insomnia  
                            - Underlying sleep disorders (e.g., sleep fragmentation disorder)  
                            """
                        )

                    # st.caption(
                    #     """
                    #     *Note: Interpretation thresholds are based on normalized model output.*
                    #     The exact thresholds depend on how the PSG model was trained.
                    #     """
                    # )


                    # st.caption(
                    #     """
                    #     This prediction is generated from:
                    #     - The EDF signal data you uploaded
                    #     - Your preprocessing pipeline in `app.utils.preprocess`
                    #     - The trained PSG model stored as `psg_model.pkl`
                    #     """
                    # )

        except Exception as e:
            st.error(f"Error processing PSG file: {e}")


# =====================================================
#  TAB 4: CORRELATION EXPLORER
# =====================================================
with tabs[3]:
    st.subheader("Correlation Explorer – ISI vs PHQ-9")

    st.write(
        """
        This section computes and visualizes the **correlation between insomnia severity**
        and **depression symptoms** using ISI and PHQ-9 scores stored in Firestore.
        """
    )

    df_corr = fetch_isi_phq9_data()

    if df_corr.empty:
        st.warning(
            "No data found in Firestore or unable to connect. "
            "Ensure the collection name and credentials are correct."
        )
    else:
        # st.markdown("#### Sample Data")
        # st.dataframe(df_corr.head())

        corr_val = df_corr[["ISI", "PHQ9"]].corr().iloc[0, 1]
        st.metric(
            label="Pearson Correlation (ISI vs PHQ-9)",
            value=f"{corr_val:.3f}",
        )

        st.markdown("#### Scatter Plot")
        st.write(
            "Each point represents a **session** with both ISI and PHQ-9 scores."
        )
        st.scatter_chart(df_corr, x="ISI", y="PHQ9")

        st.caption(
            """
            A higher positive correlation suggests that higher insomnia severity
            is associated with higher depression scores in the observed population.
            """
        )


# =====================================================
#  TAB 5: ABOUT / HOW IT WORKS
# =====================================================
# with tabs[4]:
    # st.subheader("About SleepScope & Technical Workflow")
# 
    # st.markdown(
        # """
        ## 1. Data Sources
# 
        # - **Questionnaire data**:  
        #   - Insomnia Severity Index (ISI) – severity of insomnia  
        #   - PHQ-9 – depression symptoms  
        # - **Optional PSG data** (EDF):
        #   - EEG/EOG/EMG channels extracted as features for advanced modelling.
# 
        ## 2. ML Components
# 
        # 1. **ISI-based Severity Estimation (Rule-based)**  
        #    - ISI total score is categorized into severity levels:
            #  - 0–7: No clinically significant insomnia  
            #  - 8–14: Subthreshold insomnia  
            #  - 15–21: Moderate clinical insomnia  
            #  - 22–28: Severe clinical insomnia  
# 
        # 2. **Subtype Classification (ML Model)**  
        #    - Uses `subtype_model.pkl`, `subtype_scaler.pkl`, and
            #  `subtype_features.json`.  
        #    - Features are collected from the user and scaled before prediction.  
        #    - Output label is mapped via `subtype_label_map.json`.
# 
        # 3. **Depression Correlation (Explainable Insight)**  
        #    - ISI and PHQ-9 scores are stored in Firestore along with a session ID.  
        #    - Correlation between ISI and PHQ-9 is computed and visualized to
            #  study how insomnia might co-occur with depression.
# 
        ## 3. Architecture
# 
        # - **Backend** (already deployed on Render):  
        #   - Handles model training / storage, scoring logic, and Firestore integration.
        # - **This Streamlit app** (same repo):
        #   - Acts as a lightweight, Python-based frontend.
        #   - Directly loads `.pkl` models and Firestore data.
        #   - Provides separate views for **users** and **clinicians**.
# 
        ## 4. Why Streamlit?
# 
        # - Rapid prototype for demo under strict time constraints.
        # - Eliminates complex JS–backend integration issues.
        # - Still demonstrates:
        #   - End-to-end ML workflow  
        #   - Data pipeline  
        #   - Real-time analytics & explainability  
        # """
    # )
# 