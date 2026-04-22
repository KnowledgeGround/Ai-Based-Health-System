import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import warnings
warnings.filterwarnings('ignore')

# Enhanced symptom-disease dataset (expanded for better training)
data = {
    'symptoms': [
        'fever cough fatigue',
        'chest pain shortness_of_breath sweating',
        'headache nausea vomiting sensitivity_to_light',
        'persistent_cough blood_in_cough weight_loss',
        'fever chills muscle_pain joint_pain',
        'high_fever rash neck_stiffness confusion',
        'itching skin_rash fatigue weight_gain',
        'blurred_vision increased_thirst frequent_urination',
        'abdominal_pain diarrhea vomiting dehydration',
        'swelling redness warmth pain_in_leg',
        'yellowing_of_skin nausea fatigue abdominal_pain',
        'chest_pain shortness_of_breath dizziness',
        'persistent_cough wheezing difficulty_breathing',
        'fever sore_throat swollen_lymph_nodes fatigue',
        'severe_headache vision_changes weakness_on_one_side',
        'high_fever chills sweating fatigue',
        'dry_cough fever tiredness breathing_difficulty',
        'chest_pain pressure_radiating_pain nausea',
        'irregular_heartbeat fatigue dizziness fainting',
        'memory_loss confusion disorientation personality_changes',
        'fever cough runny_nose sore_throat',
        'shortness_of_breath chest_tightness wheezing cough',
        'fever high_temperature sweating chills',
        'joint_pain swelling stiffness morning_stiffness',
        'severe_headache neck_stiffness high_fever confusion',
        'cough phlegm wheezing shortness_of_breath',
        'weight_loss persistent_cough blood_in_sputum',
        'abdominal_pain vomiting diarrhea fever',
        'skin_rash itching blister fever',
        'confusion memory_loss difficulty_concentrating'
    ],
    'disease': [
        'Common Cold', 'Heart Attack', 'Migraine', 'Lung Cancer', 'Malaria',
        'Meningitis', 'Hypothyroidism', 'Diabetes', 'Cholera', 'DVT',
        'Hepatitis', 'Pulmonary Embolism', 'Asthma', 'Mononucleosis',
        'Stroke', 'Tuberculosis', 'COVID-19', 'Heart Attack', 'Arrhythmia',
        'Alzheimers',
        'Common Cold', 'Asthma', 'Malaria', 'Rheumatoid Arthritis', 
        'Meningitis', 'Tuberculosis', 'Lung Cancer', 'Cholera',
        'Chickenpox', 'Dementia'
    ]
}

df = pd.DataFrame(data)

# Preprocessing with improved settings
vectorizer = TfidfVectorizer(max_features=150, stop_words='english', ngram_range=(1, 2))
X = vectorizer.fit_transform(df['symptoms'].str.lower())
y = df['disease']

# Split data with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("=" * 80)
print("TRAINING ENSEMBLE MODELS FOR DISEASE PREDICTION")
print("=" * 80)

# Individual models
print("\n[1/3] Training Naive Bayes...")
nb_model = MultinomialNB(alpha=0.1)
nb_model.fit(X_train, y_train)
nb_pred = nb_model.predict(X_test)
nb_acc = accuracy_score(y_test, nb_pred)
print(f"    Accuracy: {nb_acc:.2%}")

print("\n[2/3] Training Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)
print(f"    Accuracy: {rf_acc:.2%}")

print("\n[3/3] Training Gradient Boosting...")
gb_model = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
gb_model.fit(X_train, y_train)
gb_pred = gb_model.predict(X_test)
gb_acc = accuracy_score(y_test, gb_pred)
print(f"    Accuracy: {gb_acc:.2%}")

# Ensemble with weighted voting
print("\nCreating Ensemble Voting Classifier...")
model = VotingClassifier(
    estimators=[('nb', nb_model), ('rf', rf_model), ('gb', gb_model)],
    voting='soft',
    weights=[1, 2, 2]
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
ensemble_acc = accuracy_score(y_test, y_pred)
print(f"    Ensemble Accuracy: {ensemble_acc:.2%}")

# Evaluation
print("\n" + "=" * 80)
print("EVALUATION METRICS")
print("=" * 80)
print(f"\nEnsemble Model Accuracy: {ensemble_acc:.2%}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Cross-validation
cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"\nCross-Validation Accuracy: {cv_scores.mean():.2%} (+/- {cv_scores.std():.2%})")

# Save model and vectorizer
with open('symptom_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('symptom_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("\n✅ Model and vectorizer saved successfully!")