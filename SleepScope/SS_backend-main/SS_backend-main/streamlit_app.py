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
    
    /* Main container with subtle gradient mesh background */
    .main {
        background: linear-gradient(135deg, #F7F9FC 0%, #E8EAF6 50%, #F5F7FA 100%);
        padding: 2rem;
        animation: fadeIn 0.6s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    /* Title Styling with animated gradient */
    .big-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #8B7EC8 0%, #4A90E2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
        text-align: center;
        animation: gradientShift 3s ease infinite;
        background-size: 200% 200%;
    }
    
    @keyframes gradientShift {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    .subtitle {
        font-size: 1rem;
        color: #718096;
        margin-bottom: 3rem;
        text-align: center;
        font-weight: 400;
        line-height: 1.6;
    }
    
    /* Glassmorphism Card Styling */
    .stContainer {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.18);
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        margin-bottom: 1.5rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .stContainer:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.15);
    }
    
    /* Modern Button Styling with pill shape */
    .stButton>button {
        background: linear-gradient(135deg, #8B7EC8 0%, #4A90E2 100%);
        color: white;
        border: none;
        border-radius: 24px;
        padding: 0.75rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 15px rgba(139, 126, 200, 0.3);
        width: 100%;
        cursor: pointer;
    }
    
    .stButton>button:hover {
        transform: scale(1.02) translateY(-2px);
        box-shadow: 0 8px 25px rgba(139, 126, 200, 0.5);
        background: linear-gradient(135deg, #9C8DD7 0%, #5BA0F2 100%);
    }
    
    .stButton>button:active {
        transform: scale(0.98);
    }
    
    /* Metric Styling */
    .stMetric {
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(10px);
        padding: 1.5rem;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(139, 126, 200, 0.1);
        transition: all 0.3s ease;
    }
    
    .stMetric:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 24px rgba(0, 0, 0, 0.12);
    }
    
    /* Modern Tabs Styling with pill shape */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background-color: #FFFFFF;
        border-radius: 24px;
        padding: 8px;
        justify-content: center;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.06);
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 20px;
        color: #718096;
        font-weight: 600;
        padding: 12px 24px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        font-size: 1rem;
        border: none;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #F5F7FA;
        color: #4A90E2;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #8B7EC8 0%, #4A90E2 100%);
        color: white;
        box-shadow: 0 4px 12px rgba(139, 126, 200, 0.3);
    }
    
    /* Remove the default underline indicator */
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab-border"] {
        background-color: transparent;
    }
    
    /* Enhanced Alert boxes with modern styling */
    .stAlert {
        border-radius: 12px;
        border-left: 4px solid;
        padding: 1rem 1.5rem;
        backdrop-filter: blur(10px);
        animation: slideIn 0.4s ease;
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateX(-10px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    /* Select slider styling with better visual feedback */
    .stSlider {
        padding: 1rem 0;
    }
    
    .stSlider > div > div > div {
        background-color: #E8EAF6;
    }
    
    .stSlider > div > div > div > div {
        background-color: #8B7EC8;
    }
    
    /* Improved heading hierarchy */
    h1 {
        font-size: 2.25rem;
        font-weight: 700;
        color: #1A2332;
        line-height: 1.2;
        letter-spacing: -0.02em;
        margin-bottom: 1rem;
    }
    
    h2 {
        font-size: 1.75rem;
        font-weight: 600;
        color: #2D3748;
        line-height: 1.3;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    
    h3 {
        font-size: 1.25rem;
        font-weight: 600;
        color: #2D3748;
        line-height: 1.4;
        margin-top: 1.5rem;
        margin-bottom: 0.75rem;
    }
    
    h4 {
        font-size: 1.1rem;
        font-weight: 600;
        color: #4A5568;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    
    /* Body text styling */
    p {
        color: #4A5568;
        line-height: 1.6;
        font-size: 1rem;
    }
    
    /* Enhanced Input field styling with smooth focus states */
    .stNumberInput>div>div>input,
    .stSelectbox>div>div>select,
    .stTextInput>div>div>input {
        border-radius: 12px;
        border: 1px solid #E2E8F0;
        padding: 0.75rem 1rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        background-color: #FFFFFF;
        font-size: 1rem;
    }
    
    .stNumberInput>div>div>input:focus,
    .stSelectbox>div>div>select:focus,
    .stTextInput>div>div>input:focus {
        border-color: #8B7EC8;
        box-shadow: 0 0 0 3px rgba(139, 126, 200, 0.1);
        outline: none;
    }
    
    .stNumberInput>div>div>input:hover,
    .stSelectbox>div>div>select:hover,
    .stTextInput>div>div>input:hover {
        border-color: #CBD5E0;
    }
    
    /* Elegant File uploader styling */
    .stFileUploader {
        border: 2px dashed #CBD5E0;
        border-radius: 16px;
        padding: 2.5rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        background-color: rgba(255, 255, 255, 0.5);
        text-align: center;
    }
    
    .stFileUploader:hover {
        border-color: #8B7EC8;
        background-color: rgba(139, 126, 200, 0.02);
        transform: translateY(-2px);
    }
    
    /* Markdown content styling */
    .markdown-text-container {
        color: #4A5568;
        line-height: 1.7;
    }
    
    /* Elegant section divider */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #8B7EC8, #4A90E2, transparent);
        margin: 3rem 0;
        opacity: 0.6;
    }
    
    /* Success state styling */
    .stSuccess {
        background-color: rgba(72, 187, 120, 0.1);
        border-left-color: #48BB78;
    }
    
    /* Warning state styling */
    .stWarning {
        background-color: rgba(237, 137, 54, 0.1);
        border-left-color: #ED8936;
    }
    
    /* Error state styling */
    .stError {
        background-color: rgba(245, 101, 101, 0.1);
        border-left-color: #F56565;
    }
    
    /* Info state styling */
    .stInfo {
        background-color: rgba(66, 153, 225, 0.1);
        border-left-color: #4299E1;
    }
    
    /* Data table styling */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
    }
    
    /* Spinner/loading state */
    .stSpinner > div {
        border-top-color: #8B7EC8 !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: rgba(255, 255, 255, 0.8);
        border-radius: 12px;
        padding: 1rem;
        font-weight: 600;
        color: #2D3748;
        transition: all 0.3s ease;
    }
    
    .streamlit-expanderHeader:hover {
        background-color: rgba(139, 126, 200, 0.1);
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #F7F9FC;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #8B7EC8 0%, #4A90E2 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #9C8DD7 0%, #5BA0F2 100%);
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
    Load Firestore using environment variable or service account file.
    """
    try:
        if os.environ.get("FIREBASE_CREDENTIALS"):
            # Running on Render
            cred_dict = json. loads(os.environ["FIREBASE_CREDENTIALS"])
            cred = credentials.Certificate(cred_dict)
            
            if not firebase_admin._apps:
                firebase_admin.initialize_app(cred)
            
            client = firestore.client()
            return client
        else:
            # Running locally
            cred_path = os.path.join("credentials", "serviceAccount.json")
            if not os.path. exists(cred_path):
                st.error(f"Service account file not found at: {cred_path}")
                return None
                
            cred = credentials.Certificate(cred_path)
            if not firebase_admin._apps:
                firebase_admin.initialize_app(cred)
                
            client = firestore.client()
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
    
    # Enhanced Hero section with animated gradient text
    st.markdown("""
        <div style='text-align: center; padding: 3rem 0 2rem 0; animation: fadeIn 0.8s ease-in;'>
            <h1 style='font-size: 3rem; 
                       font-weight: 800;
                       background: linear-gradient(135deg, #8B7EC8 0%, #4A90E2 100%);
                       -webkit-background-clip: text;
                       -webkit-text-fill-color: transparent;
                       margin-bottom: 1.5rem;
                       letter-spacing: -0.02em;
                       line-height: 1.2;'>
                Welcome to SleepScope
            </h1>
            <p style='font-size: 1.25rem; 
                      color: #718096; 
                      max-width: 700px; 
                      margin: 0 auto;
                      line-height: 1.6;
                      font-weight: 400;'>
                An advanced explainable ML framework for comprehensive insomnia analysis and clinical decision support
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Enhanced Feature cards with glassmorphism effect
    col1, col2, col3 = st.columns(3, gap="large")
    
    with col1:
        st.markdown("""
            <div style='background: linear-gradient(135deg, #8B7EC8 0%, #4A90E2 100%); 
                        padding: 2.5rem; 
                        border-radius: 20px; 
                        color: white; 
                        text-align: center;
                        box-shadow: 0 8px 32px rgba(139, 126, 200, 0.3);
                        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                        cursor: default;'
                 onmouseover="this.style.transform='translateY(-8px) scale(1.02)'; this.style.boxShadow='0 12px 40px rgba(139, 126, 200, 0.4)'"
                 onmouseout="this.style.transform='translateY(0) scale(1)'; this.style.boxShadow='0 8px 32px rgba(139, 126, 200, 0.3)'">
                <div style='width: 70px; 
                            height: 70px; 
                            background: rgba(255,255,255,0.25); 
                            backdrop-filter: blur(10px);
                            border-radius: 50%; 
                            margin: 0 auto 1.5rem; 
                            display: flex; 
                            align-items: center; 
                            justify-content: center; 
                            font-size: 2rem;
                            font-weight: 700;
                            border: 2px solid rgba(255,255,255,0.3);'>
                    💤
                </div>
                <h3 style='color: white; margin: 0 0 0.75rem 0; font-size: 1.5rem; font-weight: 700;'>Severity Analysis</h3>
                <p style='color: rgba(255,255,255,0.95); font-size: 1rem; line-height: 1.5; margin: 0;'>
                    Clinical-grade ISI assessment with severity classification
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div style='background: linear-gradient(135deg, #8B7EC8 0%, #4A90E2 100%); 
                        padding: 2.5rem; 
                        border-radius: 20px; 
                        color: white; 
                        text-align: center;
                        box-shadow: 0 8px 32px rgba(139, 126, 200, 0.3);
                        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                        cursor: default;'
                 onmouseover="this.style.transform='translateY(-8px) scale(1.02)'; this.style.boxShadow='0 12px 40px rgba(139, 126, 200, 0.4)'"
                 onmouseout="this.style.transform='translateY(0) scale(1)'; this.style.boxShadow='0 8px 32px rgba(139, 126, 200, 0.3)'">
                <div style='width: 70px; 
                            height: 70px; 
                            background: rgba(255,255,255,0.25); 
                            backdrop-filter: blur(10px);
                            border-radius: 50%; 
                            margin: 0 auto 1.5rem; 
                            display: flex; 
                            align-items: center; 
                            justify-content: center; 
                            font-size: 2rem;
                            font-weight: 700;
                            border: 2px solid rgba(255,255,255,0.3);'>
                    🧠
                </div>
                <h3 style='color: white; margin: 0 0 0.75rem 0; font-size: 1.5rem; font-weight: 700;'>ML Subtyping</h3>
                <p style='color: rgba(255,255,255,0.95); font-size: 1rem; line-height: 1.5; margin: 0;'>
                    Advanced machine learning for precise subtype classification
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div style='background: linear-gradient(135deg, #8B7EC8 0%, #4A90E2 100%); 
                        padding: 2.5rem; 
                        border-radius: 20px; 
                        color: white; 
                        text-align: center;
                        box-shadow: 0 8px 32px rgba(139, 126, 200, 0.3);
                        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                        cursor: default;'
                 onmouseover="this.style.transform='translateY(-8px) scale(1.02)'; this.style.boxShadow='0 12px 40px rgba(139, 126, 200, 0.4)'"
                 onmouseout="this.style.transform='translateY(0) scale(1)'; this.style.boxShadow='0 8px 32px rgba(139, 126, 200, 0.3)'">
                <div style='width: 70px; 
                            height: 70px; 
                            background: rgba(255,255,255,0.25); 
                            backdrop-filter: blur(10px);
                            border-radius: 50%; 
                            margin: 0 auto 1.5rem; 
                            display: flex; 
                            align-items: center; 
                            justify-content: center; 
                            font-size: 2rem;
                            font-weight: 700;
                            border: 2px solid rgba(255,255,255,0.3);'>
                    📊
                </div>
                <h3 style='color: white; margin: 0 0 0.75rem 0; font-size: 1.5rem; font-weight: 700;'>Correlation Insights</h3>
                <p style='color: rgba(255,255,255,0.95); font-size: 1rem; line-height: 1.5; margin: 0;'>
                    Real-time ISI-PHQ9 depression correlation analytics
                </p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    # Enhanced Main content with better styling
    with st.container():
        st.markdown("""
            <div style='background: rgba(255, 255, 255, 0.7);
                        backdrop-filter: blur(10px);
                        border: 1px solid rgba(139, 126, 200, 0.1);
                        border-radius: 16px;
                        padding: 2rem;
                        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);'>
                <h3 style='color: #1A2332; margin-top: 0; font-size: 1.75rem; font-weight: 700;'>What is SleepScope?</h3>
                <p style='color: #4A5568; line-height: 1.8; font-size: 1.05rem; margin-bottom: 1rem;'>
                    <strong style='color: #2D3748;'>SleepScope</strong> is a state-of-the-art explainable machine learning framework designed to revolutionize insomnia assessment and clinical decision support.
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Feature highlights in 2 columns
        col_a, col_b = st.columns(2, gap="large")
        
        with col_a:
            st.markdown("""
                <div style='background: rgba(255, 255, 255, 0.6);
                            backdrop-filter: blur(5px);
                            border-left: 4px solid #8B7EC8;
                            padding: 1.5rem;
                            border-radius: 12px;
                            margin-bottom: 1rem;
                            box-shadow: 0 2px 12px rgba(0, 0, 0, 0.05);'>
                    <h4 style='color: #2D3748; margin-top: 0; font-size: 1.1rem; font-weight: 600;'>📋 Comprehensive Assessment</h4>
                    <p style='color: #4A5568; line-height: 1.6; margin: 0; font-size: 0.95rem;'>
                        Multi-dimensional evaluation combining ISI scores, PHQ-9 depression metrics, and optional polysomnography data
                    </p>
                </div>
                
                <div style='background: rgba(255, 255, 255, 0.6);
                            backdrop-filter: blur(5px);
                            border-left: 4px solid #4A90E2;
                            padding: 1.5rem;
                            border-radius: 12px;
                            margin-bottom: 1rem;
                            box-shadow: 0 2px 12px rgba(0, 0, 0, 0.05);'>
                    <h4 style='color: #2D3748; margin-top: 0; font-size: 1.1rem; font-weight: 600;'>🤖 ML-Driven Insights</h4>
                    <p style='color: #4A5568; line-height: 1.6; margin: 0; font-size: 0.95rem;'>
                        State-of-the-art machine learning models for accurate subtype classification and predictive analytics
                    </p>
                </div>
                
                <div style='background: rgba(255, 255, 255, 0.6);
                            backdrop-filter: blur(5px);
                            border-left: 4px solid #8B7EC8;
                            padding: 1.5rem;
                            border-radius: 12px;
                            box-shadow: 0 2px 12px rgba(0, 0, 0, 0.05);'>
                    <h4 style='color: #2D3748; margin-top: 0; font-size: 1.1rem; font-weight: 600;'>🏥 Clinical Utility</h4>
                    <p style='color: #4A5568; line-height: 1.6; margin: 0; font-size: 0.95rem;'>
                        Designed for both patient self-assessment and professional clinical decision support
                    </p>
                </div>
            """, unsafe_allow_html=True)
        
        with col_b:
            st.markdown("""
                <div style='background: rgba(255, 255, 255, 0.6);
                            backdrop-filter: blur(5px);
                            border-left: 4px solid #4A90E2;
                            padding: 1.5rem;
                            border-radius: 12px;
                            margin-bottom: 1rem;
                            box-shadow: 0 2px 12px rgba(0, 0, 0, 0.05);'>
                    <h4 style='color: #2D3748; margin-top: 0; font-size: 1.1rem; font-weight: 600;'>🔍 Explainable AI</h4>
                    <p style='color: #4A5568; line-height: 1.6; margin: 0; font-size: 0.95rem;'>
                        Transparent predictions with SHAP-based interpretability for clinical trust and validation
                    </p>
                </div>
                
                <div style='background: rgba(255, 255, 255, 0.6);
                            backdrop-filter: blur(5px);
                            border-left: 4px solid #8B7EC8;
                            padding: 1.5rem;
                            border-radius: 12px;
                            margin-bottom: 1rem;
                            box-shadow: 0 2px 12px rgba(0, 0, 0, 0.05);'>
                    <h4 style='color: #2D3748; margin-top: 0; font-size: 1.1rem; font-weight: 600;'>📈 Real-time Analytics</h4>
                    <p style='color: #4A5568; line-height: 1.6; margin: 0; font-size: 0.95rem;'>
                        Live correlation analysis using cloud-stored patient data with dynamic visualizations
                    </p>
                </div>
                
                <div style='background: rgba(255, 255, 255, 0.6);
                            backdrop-filter: blur(5px);
                            border-left: 4px solid #4A90E2;
                            padding: 1.5rem;
                            border-radius: 12px;
                            box-shadow: 0 2px 12px rgba(0, 0, 0, 0.05);'>
                    <h4 style='color: #2D3748; margin-top: 0; font-size: 1.1rem; font-weight: 600;'>🎯 Unified Interface</h4>
                    <p style='color: #4A5568; line-height: 1.6; margin: 0; font-size: 0.95rem;'>
                        Integrated dashboard serving both end-users and healthcare professionals seamlessly
                    </p>
                </div>
            """, unsafe_allow_html=True)

# =====================================================
#  TAB 2: USER DASHBOARD (with full ISI + PHQ-9 questionnaires)
# =====================================================
with tabs[1]:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
        <div style='text-align: center; margin-bottom: 3rem; animation: fadeIn 0.6s ease-in;'>
            <h1 style='color: #1A2332; 
                       font-size: 2.25rem; 
                       font-weight: 700;
                       margin-bottom: 0.75rem;
                       letter-spacing: -0.01em;'>
                User Assessment Dashboard
            </h1>
            <p style='color: #718096; 
                      font-size: 1.1rem; 
                      line-height: 1.6;
                      max-width: 700px;
                      margin: 0 auto;'>
                Complete the ISI and PHQ-9 questionnaires for personalized insomnia analysis and subtype classification
            </p>
        </div>
    """, unsafe_allow_html=True)

    # -------------------------
    # ISI QUESTIONS (0–4)
    # -------------------------
    st.markdown("""
        <div style='background: linear-gradient(135deg, #8B7EC8 0%, #4A90E2 100%); 
                    padding: 2rem; 
                    border-radius: 16px; 
                    margin-bottom: 2rem;
                    box-shadow: 0 4px 20px rgba(139, 126, 200, 0.2);'>
            <div style='display: flex; align-items: center; justify-content: space-between;'>
                <div>
                    <h2 style='color: white; margin: 0; font-size: 1.75rem; font-weight: 700;'>
                        💤 Insomnia Severity Index (ISI)
                    </h2>
                    <p style='color: rgba(255,255,255,0.95); margin: 0.75rem 0 0 0; font-size: 1rem;'>
                        Answer each question based on your sleep patterns over the past two weeks
                    </p>
                </div>
                <div style='background: rgba(255,255,255,0.25);
                            backdrop-filter: blur(10px);
                            padding: 0.75rem 1.5rem;
                            border-radius: 12px;
                            border: 1px solid rgba(255,255,255,0.3);'>
                    <span style='color: white; font-size: 0.9rem; font-weight: 600;'>Score Range: 0–28</span>
                </div>
            </div>
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
        <div style='background: rgba(139, 126, 200, 0.1); 
                    padding: 1.5rem; 
                    border-radius: 12px; 
                    border-left: 4px solid #8B7EC8; 
                    margin: 1.5rem 0;
                    box-shadow: 0 2px 12px rgba(139, 126, 200, 0.1);'>
            <div style='display: flex; align-items: center; justify-content: space-between;'>
                <strong style='color: #1A2332; font-size: 1.1rem; font-weight: 600;'>Total ISI Score</strong>
                <span style='background: linear-gradient(135deg, #8B7EC8 0%, #4A90E2 100%);
                             color: white;
                             padding: 0.5rem 1.25rem;
                             border-radius: 20px;
                             font-size: 1.5rem;
                             font-weight: 700;
                             box-shadow: 0 2px 8px rgba(139, 126, 200, 0.3);'>
                    {isi_total}/28
                </span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # -------------------------
    # PHQ-9 QUESTIONS (0–27)
    # -------------------------
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
        <div style='background: linear-gradient(135deg, #8B7EC8 0%, #4A90E2 100%); 
                    padding: 2rem; 
                    border-radius: 16px; 
                    margin-bottom: 2rem;
                    box-shadow: 0 4px 20px rgba(139, 126, 200, 0.2);'>
            <div style='display: flex; align-items: center; justify-content: space-between;'>
                <div>
                    <h2 style='color: white; margin: 0; font-size: 1.75rem; font-weight: 700;'>
                        🧠 Patient Health Questionnaire (PHQ-9)
                    </h2>
                    <p style='color: rgba(255,255,255,0.95); margin: 0.75rem 0 0 0; font-size: 1rem;'>
                        Over the last two weeks, how often have you been bothered by the following problems?
                    </p>
                </div>
                <div style='background: rgba(255,255,255,0.25);
                            backdrop-filter: blur(10px);
                            padding: 0.75rem 1.5rem;
                            border-radius: 12px;
                            border: 1px solid rgba(255,255,255,0.3);'>
                    <span style='color: white; font-size: 0.9rem; font-weight: 600;'>Score Range: 0–27</span>
                </div>
            </div>
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
        <div style='background: rgba(139, 126, 200, 0.1); 
                    padding: 1.5rem; 
                    border-radius: 12px; 
                    border-left: 4px solid #4A90E2; 
                    margin: 1.5rem 0;
                    box-shadow: 0 2px 12px rgba(74, 144, 226, 0.1);'>
            <div style='display: flex; align-items: center; justify-content: space-between;'>
                <strong style='color: #1A2332; font-size: 1.1rem; font-weight: 600;'>Total PHQ-9 Score</strong>
                <span style='background: linear-gradient(135deg, #8B7EC8 0%, #4A90E2 100%);
                             color: white;
                             padding: 0.5rem 1.25rem;
                             border-radius: 20px;
                             font-size: 1.5rem;
                             font-weight: 700;
                             box-shadow: 0 2px 8px rgba(74, 144, 226, 0.3);'>
                    {phq9_total}/27
                </span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # ---------------------------------------------------------
    # Subtype Feature Questions (clean, user-friendly inputs)
    # ---------------------------------------------------------
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
        <div style='background: linear-gradient(135deg, #8B7EC8 0%, #4A90E2 100%); 
                    padding: 2rem; 
                    border-radius: 16px; 
                    margin-bottom: 2rem;
                    box-shadow: 0 4px 20px rgba(139, 126, 200, 0.2);'>
            <div style='display: flex; align-items: center; justify-content: space-between;'>
                <div>
                    <h2 style='color: white; margin: 0; font-size: 1.75rem; font-weight: 700;'>
                        🌙 Sleep & Lifestyle Assessment
                    </h2>
                    <p style='color: rgba(255,255,255,0.95); margin: 0.75rem 0 0 0; font-size: 1rem;'>
                        Additional information for accurate subtype classification
                    </p>
                </div>
            </div>
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
        
        # Determine severity color and emoji based on score
        if isi_total <= 7:
            severity_color = "#48BB78"  # Green
            severity_emoji = "✅"
            severity_bg = "rgba(72, 187, 120, 0.1)"
        elif isi_total <= 14:
            severity_color = "#4299E1"  # Blue
            severity_emoji = "ℹ️"
            severity_bg = "rgba(66, 153, 225, 0.1)"
        elif isi_total <= 21:
            severity_color = "#ED8936"  # Orange
            severity_emoji = "⚠️"
            severity_bg = "rgba(237, 137, 54, 0.1)"
        else:
            severity_color = "#F56565"  # Red
            severity_emoji = "🚨"
            severity_bg = "rgba(245, 101, 101, 0.1)"

        colA, colB = st.columns(2, gap="large")

        with colA:
            st.markdown(f"""
                <div style='background: {severity_bg}; 
                            backdrop-filter: blur(10px);
                            padding: 2.5rem; 
                            border-radius: 16px; 
                            text-align: center;
                            border: 2px solid {severity_color}40;
                            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.08);
                            transition: all 0.3s ease;'>
                    <div style='font-size: 3rem; margin-bottom: 1rem;'>{severity_emoji}</div>
                    <h3 style='color: #1A2332; margin: 0 0 1rem 0; font-size: 1.3rem; font-weight: 700;'>
                        Insomnia Severity
                    </h3>
                    <div style='background: {severity_color};
                                color: white;
                                padding: 1rem 1.5rem;
                                border-radius: 12px;
                                margin: 1rem 0;
                                font-size: 1.25rem;
                                font-weight: 600;
                                box-shadow: 0 4px 12px {severity_color}40;'>
                        {severity_label}
                    </div>
                    <p style='color: #4A5568; margin: 1rem 0 0 0; font-size: 1rem;'>
                        ISI Score: <strong style='color: {severity_color};'>{isi_total}/28</strong>
                    </p>
                </div>
            """, unsafe_allow_html=True)

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
                <div style='background: linear-gradient(135deg, rgba(139, 126, 200, 0.15) 0%, rgba(74, 144, 226, 0.15) 100%); 
                            backdrop-filter: blur(10px);
                            padding: 2.5rem; 
                            border-radius: 16px; 
                            text-align: center;
                            border: 2px solid rgba(139, 126, 200, 0.3);
                            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.08);
                            transition: all 0.3s ease;'>
                    <div style='font-size: 3rem; margin-bottom: 1rem;'>🧬</div>
                    <h3 style='color: #1A2332; margin: 0 0 1rem 0; font-size: 1.3rem; font-weight: 700;'>
                        Insomnia Subtype
                    </h3>
                    <div style='background: linear-gradient(135deg, #8B7EC8 0%, #4A90E2 100%);
                                color: white;
                                padding: 1rem 1.5rem;
                                border-radius: 12px;
                                margin: 1rem 0;
                                font-size: 1.25rem;
                                font-weight: 600;
                                box-shadow: 0 4px 12px rgba(139, 126, 200, 0.4);'>
                        {pretty_label}
                    </div>
                    <p style='color: #4A5568; margin: 1rem 0 0 0; font-size: 0.95rem;'>
                        Model Classification: <strong style='color: #8B7EC8;'>{raw_pred}</strong>
                    </p>
                </div>
            """, unsafe_allow_html=True)


# =====================================================
#  TAB 3: CLINICIAN (PSG + HYPNOGRAM UPLOAD)
# =====================================================
with tabs[2]:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
        <div style='text-align: center; margin-bottom: 3rem; animation: fadeIn 0.6s ease-in;'>
            <h1 style='color: #1A2332; 
                       font-size: 2.25rem; 
                       font-weight: 700;
                       margin-bottom: 0.75rem;
                       letter-spacing: -0.01em;'>
                Clinician Dashboard – PSG Analysis
            </h1>
            <p style='color: #718096; 
                      font-size: 1.1rem; 
                      line-height: 1.6;
                      max-width: 700px;
                      margin: 0 auto;'>
                Upload polysomnography data for advanced ML-based analysis with SHAP explainability
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div style='background: rgba(139, 126, 200, 0.1);
                    backdrop-filter: blur(5px);
                    border-left: 4px solid #8B7EC8;
                    padding: 1.25rem 1.5rem;
                    border-radius: 12px;
                    margin-bottom: 2rem;'>
            <p style='color: #2D3748; margin: 0; font-size: 1rem; line-height: 1.6;'>
                <strong>📁 Required Files:</strong> Please upload both the PSG file and its corresponding Hypnogram in EDF format
            </p>
        </div>
    """, unsafe_allow_html=True)

    col_psg, col_hyp = st.columns(2, gap="large")

    with col_psg:
        st.markdown("""
            <div style='text-align: center; margin-bottom: 1rem;'>
                <h4 style='color: #2D3748; font-size: 1.1rem; font-weight: 600; margin: 0;'>
                    📊 PSG / EDF File
                </h4>
            </div>
        """, unsafe_allow_html=True)
        uploaded_psg = st.file_uploader(
            "Upload PSG File",
            type=["edf", "EDF"],
            key="psg_file",
            label_visibility="collapsed"
        )

    with col_hyp:
        st.markdown("""
            <div style='text-align: center; margin-bottom: 1rem;'>
                <h4 style='color: #2D3748; font-size: 1.1rem; font-weight: 600; margin: 0;'>
                    📈 Hypnogram / EDF File
                </h4>
            </div>
        """, unsafe_allow_html=True)
        uploaded_hyp = st.file_uploader(
            "Upload Hypnogram File",
            type=["edf", "EDF"],
            key="hyp_file",
            label_visibility="collapsed"
        )

    if uploaded_psg and uploaded_hyp:
        st.markdown(f"""
            <div style='background: rgba(72, 187, 120, 0.1);
                        backdrop-filter: blur(5px);
                        border-left: 4px solid #48BB78;
                        padding: 1.5rem;
                        border-radius: 12px;
                        margin: 1.5rem 0;'>
                <div style='display: flex; align-items: center; gap: 0.75rem;'>
                    <span style='font-size: 1.5rem;'>✅</span>
                    <div>
                        <strong style='color: #1A2332; font-size: 1.1rem;'>Files Successfully Uploaded</strong>
                        <p style='color: #4A5568; margin: 0.5rem 0 0 0; font-size: 0.95rem;'>
                            • PSG: <strong>{uploaded_psg.name}</strong><br>
                            • Hypnogram: <strong>{uploaded_hyp.name}</strong>
                        </p>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # Write to temp files
        with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp_psg:
            tmp_psg.write(uploaded_psg.read())
            psg_path = tmp_psg.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp_hyp:
            tmp_hyp.write(uploaded_hyp.read())
            hyp_path = tmp_hyp.name

        st.markdown("""
            <div style='background: rgba(74, 144, 226, 0.1);
                        backdrop-filter: blur(5px);
                        border-left: 4px solid #4A90E2;
                        padding: 1.25rem 1.5rem;
                        border-radius: 12px;
                        margin: 1.5rem 0;'>
                <p style='color: #2D3748; margin: 0; font-size: 1rem;'>
                    <strong>⏳ Processing:</strong> Extracting PSG features from uploaded files...
                </p>
            </div>
        """, unsafe_allow_html=True)

        try:
            # Extract features (your original function)
            psg_features = extract_psg_features(psg_path, hyp_path)

            if psg_features is None:
                st.markdown("""
                    <div style='background: rgba(245, 101, 101, 0.1);
                                backdrop-filter: blur(5px);
                                border-left: 4px solid #F56565;
                                padding: 1.5rem;
                                border-radius: 12px;
                                margin: 1.5rem 0;'>
                        <div style='display: flex; align-items: center; gap: 0.75rem;'>
                            <span style='font-size: 1.5rem;'>❌</span>
                            <div>
                                <strong style='color: #1A2332; font-size: 1.1rem;'>Feature Extraction Failed</strong>
                                <p style='color: #4A5568; margin: 0.5rem 0 0 0; font-size: 0.95rem;'>
                                    No values were returned from the feature extraction process.
                                </p>
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div style='background: rgba(72, 187, 120, 0.1);
                                backdrop-filter: blur(5px);
                                border-left: 4px solid #48BB78;
                                padding: 1.5rem;
                                border-radius: 12px;
                                margin: 1.5rem 0;'>
                        <div style='display: flex; align-items: center; gap: 0.75rem;'>
                            <span style='font-size: 1.5rem;'>✅</span>
                            <div>
                                <strong style='color: #1A2332; font-size: 1.1rem;'>Feature Extraction Complete</strong>
                                <p style='color: #4A5568; margin: 0.5rem 0 0 0; font-size: 0.95rem;'>
                                    PSG features successfully extracted and ready for analysis
                                </p>
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                

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

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("""
                    <div style='text-align: center; margin: 2rem 0 1.5rem 0;'>
                        <h2 style='color: #1A2332; font-size: 1.75rem; font-weight: 700; margin: 0;'>
                            🔬 PSG Model Prediction
                        </h2>
                    </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                    <div style='background: linear-gradient(135deg, rgba(139, 126, 200, 0.15) 0%, rgba(74, 144, 226, 0.15) 100%);
                                backdrop-filter: blur(10px);
                                padding: 2.5rem;
                                border-radius: 16px;
                                text-align: center;
                                border: 2px solid rgba(139, 126, 200, 0.3);
                                box-shadow: 0 8px 24px rgba(0, 0, 0, 0.08);
                                margin: 1rem 0 2rem 0;'>
                        <h3 style='color: #2D3748; margin: 0 0 1.5rem 0; font-size: 1.2rem; font-weight: 600;'>
                            Predicted Insomnia Severity Score
                        </h3>
                        <div style='background: linear-gradient(135deg, #8B7EC8 0%, #4A90E2 100%);
                                    color: white;
                                    padding: 1.5rem 2rem;
                                    border-radius: 16px;
                                    font-size: 3rem;
                                    font-weight: 700;
                                    box-shadow: 0 6px 20px rgba(139, 126, 200, 0.4);
                                    display: inline-block;
                                    min-width: 200px;'>
                            {prediction:.2f}
                        </div>
                        <p style='color: #4A5568; margin: 1.5rem 0 0 0; font-size: 1rem;'>
                            Based on extracted PSG features and ML model analysis
                        </p>
                    </div>
                """, unsafe_allow_html=True)

                # -----------------------
                # SHAP XAI
                # -----------------------
                st.markdown("""
                    <div style='text-align: center; margin: 3rem 0 2rem 0;'>
                        <h2 style='color: #1A2332; font-size: 1.75rem; font-weight: 700; margin: 0;'>
                            🔍 Explainable AI Interpretation
                        </h2>
                        <p style='color: #718096; margin: 0.75rem 0 0 0; font-size: 1rem;'>
                            SHAP-based feature importance and contribution analysis
                        </p>
                    </div>
                """, unsafe_allow_html=True)

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
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
        <div style='text-align: center; margin-bottom: 3rem; animation: fadeIn 0.6s ease-in;'>
            <h1 style='color: #1A2332; 
                       font-size: 2.25rem; 
                       font-weight: 700;
                       margin-bottom: 0.75rem;
                       letter-spacing: -0.01em;'>
                Correlation Explorer
            </h1>
            <p style='color: #718096; 
                      font-size: 1.1rem; 
                      line-height: 1.6;
                      max-width: 700px;
                      margin: 0 auto;'>
                Analyze the relationship between insomnia severity and depression symptoms across patient data
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div style='background: rgba(139, 126, 200, 0.1);
                    backdrop-filter: blur(5px);
                    border-left: 4px solid #8B7EC8;
                    padding: 1.25rem 1.5rem;
                    border-radius: 12px;
                    margin-bottom: 2rem;'>
            <p style='color: #2D3748; margin: 0; font-size: 1rem; line-height: 1.6;'>
                <strong>📊 Data Analysis:</strong> Real-time correlation between ISI and PHQ-9 scores from cloud-stored patient sessions
            </p>
        </div>
    """, unsafe_allow_html=True)

    df_corr = fetch_isi_phq9_data()

    if df_corr.empty:
        st.markdown("""
            <div style='background: rgba(237, 137, 54, 0.1);
                        backdrop-filter: blur(5px);
                        border-left: 4px solid #ED8936;
                        padding: 2rem;
                        border-radius: 12px;
                        text-align: center;
                        margin: 2rem 0;'>
                <div style='font-size: 3rem; margin-bottom: 1rem;'>📭</div>
                <h3 style='color: #1A2332; margin: 0 0 0.75rem 0; font-size: 1.3rem; font-weight: 600;'>
                    No Data Available
                </h3>
                <p style='color: #4A5568; margin: 0; font-size: 1rem; line-height: 1.6;'>
                    No patient session data found in Firestore. Complete assessments in the User Dashboard to populate this analysis.
                </p>
            </div>
        """, unsafe_allow_html=True)
    else:
        # st.markdown("#### Sample Data")
        # st.dataframe(df_corr.head())

        corr_val = df_corr[["ISI", "PHQ9"]].corr().iloc[0, 1]
        
        # Determine correlation strength
        if abs(corr_val) >= 0.7:
            corr_strength = "Strong"
            corr_color = "#F56565"
            corr_emoji = "🔴"
        elif abs(corr_val) >= 0.4:
            corr_strength = "Moderate"
            corr_color = "#ED8936"
            corr_emoji = "🟠"
        else:
            corr_strength = "Weak"
            corr_color = "#4299E1"
            corr_emoji = "🔵"
        
        # Correlation metric card
        st.markdown(f"""
            <div style='background: linear-gradient(135deg, rgba(139, 126, 200, 0.15) 0%, rgba(74, 144, 226, 0.15) 100%);
                        backdrop-filter: blur(10px);
                        padding: 2.5rem;
                        border-radius: 16px;
                        text-align: center;
                        border: 2px solid rgba(139, 126, 200, 0.3);
                        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.08);
                        margin: 2rem 0;'>
                <h3 style='color: #2D3748; margin: 0 0 1.5rem 0; font-size: 1.2rem; font-weight: 600;'>
                    📈 Pearson Correlation Coefficient
                </h3>
                <div style='display: flex; align-items: center; justify-content: center; gap: 2rem; flex-wrap: wrap;'>
                    <div>
                        <div style='background: linear-gradient(135deg, #8B7EC8 0%, #4A90E2 100%);
                                    color: white;
                                    padding: 1.5rem 2rem;
                                    border-radius: 16px;
                                    font-size: 3rem;
                                    font-weight: 700;
                                    box-shadow: 0 6px 20px rgba(139, 126, 200, 0.4);
                                    min-width: 150px;'>
                            {corr_val:.3f}
                        </div>
                        <p style='color: #4A5568; margin: 1rem 0 0 0; font-size: 0.95rem;'>
                            ISI vs PHQ-9
                        </p>
                    </div>
                    <div style='text-align: left;'>
                        <div style='background: {corr_color}20;
                                    border: 2px solid {corr_color};
                                    color: {corr_color};
                                    padding: 0.75rem 1.5rem;
                                    border-radius: 12px;
                                    font-size: 1.1rem;
                                    font-weight: 600;
                                    margin-bottom: 0.5rem;'>
                            {corr_emoji} {corr_strength} Correlation
                        </div>
                        <p style='color: #4A5568; margin: 0; font-size: 0.9rem;'>
                            Based on {len(df_corr)} patient sessions
                        </p>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("""
            <div style='text-align: center; margin: 2rem 0 1.5rem 0;'>
                <h2 style='color: #1A2332; font-size: 1.5rem; font-weight: 700; margin: 0;'>
                    Scatter Plot Analysis
                </h2>
                <p style='color: #718096; margin: 0.75rem 0 0 0; font-size: 1rem;'>
                    Each point represents a patient session with both ISI and PHQ-9 scores
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # Scatter chart with styling container
        st.markdown("""
            <div style='background: rgba(255, 255, 255, 0.8);
                        backdrop-filter: blur(10px);
                        border-radius: 16px;
                        padding: 2rem;
                        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
                        border: 1px solid rgba(139, 126, 200, 0.1);'>
        """, unsafe_allow_html=True)
        
        st.scatter_chart(df_corr, x="ISI", y="PHQ9")
        
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(f"""
            <div style='background: rgba(66, 153, 225, 0.1);
                        backdrop-filter: blur(5px);
                        border-left: 4px solid #4299E1;
                        padding: 1.25rem 1.5rem;
                        border-radius: 12px;
                        margin: 1.5rem 0;'>
                <p style='color: #2D3748; margin: 0; font-size: 0.95rem; line-height: 1.7;'>
                    <strong>💡 Insight:</strong> A correlation coefficient of <strong>{corr_val:.3f}</strong> suggests that 
                    {'higher insomnia severity is moderately to strongly associated with higher depression scores' if corr_val > 0.4 
                     else 'there is a relationship between insomnia severity and depression symptoms'} 
                    in the observed patient population.
                </p>
            </div>
        """, unsafe_allow_html=True)


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
