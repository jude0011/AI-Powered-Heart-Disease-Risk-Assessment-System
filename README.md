# AI-Powered-Heart-Disease-Risk-Assessment-System
🩺 Cardiac Intelligence Portal
📖 Overview
The Cardiac Intelligence Portal is a web-based clinical decision support tool designed to estimate a patient’s risk of heart disease using machine learning.
It combines:
•	Predictive analytics
•	Explainable AI (XAI)
•	User-friendly clinical interface

The goal is simple:
👉 Help healthcare professionals and users understand not just the risk, but why the risk exists.

🎯 Key Features
1. 🧾 Patient Data Input (Interactive UI)

Users enter patient information through a structured sidebar, grouped into:

•	Demographics (Age, Sex, Thalassemia)
•	Clinical Vitals (Blood Pressure, Cholesterol, Blood Sugar)
•	Cardiac Diagnostics (ECG, Heart Rate, Chest Pain, etc.)

This makes the app intuitive and easy to use—even for non-technical users.

2. 🤖 AI Risk Prediction Engine

The system uses a trained machine learning model (Calibrated Random Forest) to:

•	Analyze patient data
•	Output a probability score (%) for heart disease

👉 Example output:

•	LOW Risk
•	HIGH Risk
•	CRITICAL Risk
3. ⚠️ Data Reliability Check (OOD Detection)

The app includes an Out-of-Distribution (OOD) Detector that checks if the input data is unusual or outside the model’s experience.

•	✅ High Reliability → Prediction is trustworthy
•	⚠️ Low Reliability → Use caution (data is rare/unusual)

This is a powerful feature often missing in basic AI apps.

4. 🧠 Explainable AI (SHAP Integration)

Instead of a “black box,” the app explains predictions using SHAP (Shapley Additive Explanations).

What it shows:
•	🔴 Features increasing risk
•	🟢 Features reducing risk
•	📊 Visual breakdown of feature impact

👉 Example:

•	“High cholesterol is increasing cardiac risk”
•	“Low ST depression is reducing risk”
5. 📊 Visual Explanation Dashboard

A SHAP Waterfall Chart visually explains:

How each feature contributes to the final prediction
The path from baseline risk → final risk score

Supports both:

•	🌙 Dark mode
•	☀️ Light mode
6. 🏥 Clinical Decision Support

Based on the predicted risk, the app provides actionable recommendations:

•	Critical Risk → Urgent cardiology referral
•	Moderate Risk → Further diagnostic testing
•	Low Risk → Preventive care
7. 📥 Report Export

Users can download a simple text report including:

•	Risk score
•	Data reliability status
8. 🔍 System Transparency

The app openly shows:

•	Model type
•	Explainability method
•	Dataset source

This builds trust and credibility, especially in healthcare applications.

⚙️ System Architecture

The app is built using the following components:

Frontend
•	Streamlit (interactive web interface)
Backend / AI Engine
•	Machine Learning Model (Random Forest)
•	Scaler (data normalization)
•	OOD Detector (anomaly detection)
Explainability Layer
•	SHAP Explainer
Visualization
•	Matplotlib (charts)
•	Custom CSS (modern UI styling)

🔄 Workflow
1.	User inputs patient data
2.	Data is scaled and processed
3.	Model predicts heart disease risk
4.	OOD detector checks data reliability
5.	SHAP explains the prediction
6.	Results + recommendations are displayed
7.	Optional: report download
🎨 User Experience Design

The app focuses heavily on usability:

•	Clean dashboard layout
•	Sidebar-based input system
•	Color-coded risk indicators
•	Dark/light theme compatibility
•	Simple, readable explanations
🚀 Use Cases

This system can be used in:

•	Hospitals and clinics
•	Telemedicine platforms
•	Health-tech startups
•	Academic research projects
•	Personal health monitoring tools
⚠️ Limitations
•	Not a replacement for professional medical diagnosis
•	Depends on training dataset (UCI Cleveland dataset)
•	No real-time hospital integration yet
•	Limited to selected clinical features
🔮 Future Improvements
•	Add biometric authentication integration (aligning with your MFA project 👀)
•	Deploy as a full API backend
•	Integrate with hospital databases
•	Add mobile app version
•	Include more advanced deep learning models
🧾 Conclusion

The Cardiac Intelligence Portal is more than just a prediction tool—it’s an intelligent, explainable, and user-friendly healthcare assistant.

It bridges the gap between:

•	AI predictions
•	Clinical understanding
•	User trust
[judeokeke/Cardiac-Risk-AI](https://huggingface.co/spaces/judeokeke/Cardiac-Risk-AI)
