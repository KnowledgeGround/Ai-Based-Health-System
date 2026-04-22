from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import numpy as np
import os
import requests
import threading
import time

app = Flask(__name__)
CORS(app)

try:
    with open('symptom_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('symptom_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    print("✅ Model loaded successfully!")
except FileNotFoundError:
    print("❌ Model files not found. Please run train_model.py first!")
    model = None
    vectorizer = None

SERIOUS_CONDITIONS = {
    'Heart Attack', 'Stroke', 'Lung Cancer', 'Meningitis', 
    'Pulmonary Embolism', 'Tuberculosis', 'Alzheimers'
}

DISEASE_INFO = {
    'Common Cold': {
        'description': 'A viral infection of the upper respiratory tract that primarily affects the nose and throat.',
        'duration': '7-10 days',
        'risk_level': 'Low',
        'recommendations': [
            'Rest and stay hydrated',
            'Use over-the-counter cold medications',
            'Use saline nasal sprays',
            'Avoid close contact with others',
            'Wash hands frequently'
        ]
    },
    'Flu': {
        'description': 'A contagious respiratory illness caused by influenza viruses that infect the nose, throat, and sometimes the lungs.',
        'duration': '1-2 weeks',
        'risk_level': 'Moderate',
        'recommendations': [
            'Get plenty of rest',
            'Stay hydrated with fluids',
            'Take antiviral medications if prescribed',
            'Use pain relievers for fever and aches',
            'Avoid spreading to others'
        ]
    },
    'COVID-19': {
        'description': 'A respiratory illness caused by the SARS-CoV-2 virus, characterized by fever, cough, and difficulty breathing.',
        'duration': '2-4 weeks (varies by severity)',
        'risk_level': 'High',
        'recommendations': [
            'Isolate immediately and contact healthcare provider',
            'Monitor symptoms closely',
            'Use prescribed treatments',
            'Wear mask and practice social distancing',
            'Get tested and follow local health guidelines'
        ]
    },
    'Pneumonia': {
        'description': 'An infection that inflames the air sacs in one or both lungs, which may fill with fluid or pus.',
        'duration': '1-3 weeks with treatment',
        'risk_level': 'High',
        'recommendations': [
            'Seek immediate medical attention',
            'Take prescribed antibiotics',
            'Rest and stay hydrated',
            'Use oxygen therapy if needed',
            'Follow up with healthcare provider'
        ]
    },
    'Bronchitis': {
        'description': 'Inflammation of the lining of the bronchial tubes, which carry air to and from the lungs.',
        'duration': '1-3 weeks',
        'risk_level': 'Moderate',
        'recommendations': [
            'Rest and drink plenty of fluids',
            'Use humidifiers to ease breathing',
            'Take cough suppressants if needed',
            'Avoid irritants like smoke',
            'See doctor if symptoms worsen'
        ]
    },
    'Asthma': {
        'description': 'A chronic respiratory condition characterized by inflammation and narrowing of the airways.',
        'duration': 'Chronic condition',
        'risk_level': 'Moderate to High',
        'recommendations': [
            'Use prescribed inhalers regularly',
            'Identify and avoid triggers',
            'Create an asthma action plan',
            'Monitor peak flow readings',
            'Seek emergency care for severe attacks'
        ]
    },
    'Heart Attack': {
        'description': 'A blockage of blood flow to the heart muscle, causing damage to the heart tissue.',
        'duration': 'Recovery takes weeks to months',
        'risk_level': 'Critical',
        'recommendations': [
            'Call emergency services immediately',
            'Chew aspirin if not allergic',
            'Stay calm and rest',
            'CPR if person is unconscious',
            'Follow cardiac rehabilitation program'
        ]
    },
    'Stroke': {
        'description': 'A medical emergency that occurs when blood flow to the brain is interrupted.',
        'duration': 'Recovery varies, often lifelong effects',
        'risk_level': 'Critical',
        'recommendations': [
            'Call emergency services immediately',
            'Note time symptoms started',
            'Do not give food or drink',
            'Loosen tight clothing',
            'Begin rehabilitation therapy as prescribed'
        ]
    },
    'Diabetes': {
        'description': 'A metabolic disorder characterized by high blood sugar levels over a prolonged period.',
        'duration': 'Chronic lifelong condition',
        'risk_level': 'Moderate to High',
        'recommendations': [
            'Monitor blood sugar regularly',
            'Take medications as prescribed',
            'Follow a healthy diet',
            'Exercise regularly',
            'Regular check-ups with healthcare provider'
        ]
    },
    'Hypertension': {
        'description': 'High blood pressure that can lead to serious health problems if not managed.',
        'duration': 'Chronic condition',
        'risk_level': 'Moderate',
        'recommendations': [
            'Take blood pressure medications regularly',
            'Reduce salt intake',
            'Maintain healthy weight',
            'Exercise regularly',
            'Limit alcohol consumption'
        ]
    },
    'Migraine': {
        'description': 'A neurological condition characterized by intense, debilitating headaches often accompanied by nausea and sensitivity to light.',
        'duration': '4-72 hours',
        'risk_level': 'Moderate',
        'recommendations': [
            'Rest in a dark, quiet room',
            'Apply cold or warm compresses',
            'Take prescribed migraine medications',
            'Identify and avoid triggers',
            'Keep a headache diary'
        ]
    },
    'Gastritis': {
        'description': 'Inflammation of the stomach lining that can cause pain, nausea, and other digestive symptoms.',
        'duration': 'Acute: days to weeks, Chronic: ongoing',
        'risk_level': 'Moderate',
        'recommendations': [
            'Avoid irritants like alcohol and NSAIDs',
            'Eat smaller, more frequent meals',
            'Take acid-reducing medications',
            'Manage stress levels',
            'Follow up with gastroenterologist'
        ]
    },
    'Food Poisoning': {
        'description': 'Illness caused by eating contaminated food, leading to gastrointestinal symptoms.',
        'duration': '24-72 hours',
        'risk_level': 'Moderate',
        'recommendations': [
            'Stay hydrated with clear fluids',
            'Rest and avoid solid foods initially',
            'Use anti-nausea medications if needed',
            'Contact doctor if severe symptoms',
            'Practice food safety measures'
        ]
    },
    'Allergies': {
        'description': 'An immune system response to substances that are typically harmless to most people.',
        'duration': 'Seasonal or year-round depending on allergen',
        'risk_level': 'Low to Moderate',
        'recommendations': [
            'Identify and avoid allergens',
            'Use antihistamine medications',
            'Consider allergy shots for severe cases',
            'Keep allergy diary',
            'Use air purifiers and mattress covers'
        ]
    },
    'Anxiety': {
        'description': 'A mental health condition characterized by feelings of worry, nervousness, or fear.',
        'duration': 'Varies, can be chronic',
        'risk_level': 'Moderate',
        'recommendations': [
            'Practice relaxation techniques',
            'Exercise regularly',
            'Consider therapy or counseling',
            'Limit caffeine and alcohol',
            'Use prescribed medications if needed'
        ]
    },
    'Depression': {
        'description': 'A mood disorder characterized by persistent feelings of sadness and loss of interest.',
        'duration': 'Varies, can be chronic',
        'risk_level': 'Moderate to High',
        'recommendations': [
            'Seek professional help',
            'Exercise regularly',
            'Maintain social connections',
            'Practice good sleep habits',
            'Consider therapy and/or medication'
        ]
    },
    'Arthritis': {
        'description': 'Inflammation of one or more joints, causing pain and stiffness.',
        'duration': 'Chronic condition',
        'risk_level': 'Moderate',
        'recommendations': [
            'Take prescribed anti-inflammatory medications',
            'Exercise regularly to maintain mobility',
            'Use heat/cold therapy',
            'Maintain healthy weight',
            'Consider physical therapy'
        ]
    },
    'Mononucleosis': {
        'description': 'An infectious illness caused by the Epstein-Barr virus, often called "mono" or "the kissing disease."',
        'duration': '2-4 weeks for acute phase, fatigue may last months',
        'risk_level': 'Moderate',
        'recommendations': [
            'Rest and avoid strenuous activity',
            'Stay hydrated and eat nutritious foods',
            'Take pain relievers for fever and sore throat',
            'Avoid contact sports during recovery',
            'Gradually return to normal activities'
        ]
    },
    'Lung Cancer': {
        'description': 'Cancer that begins in the lungs and can spread to other parts of the body.',
        'duration': 'Varies by stage and treatment',
        'risk_level': 'Critical',
        'recommendations': [
            'Seek immediate oncology consultation',
            'Consider chemotherapy, radiation, or surgery',
            'Join support groups',
            'Quit smoking if applicable',
            'Follow treatment plan closely'
        ]
    },
    'Meningitis': {
        'description': 'Inflammation of the membranes surrounding the brain and spinal cord.',
        'duration': 'Varies by type and treatment',
        'risk_level': 'Critical',
        'recommendations': [
            'Seek emergency medical care immediately',
            'Receive appropriate antibiotic/antiviral treatment',
            'Isolate to prevent spread',
            'Monitor for complications',
            'Complete full course of treatment'
        ]
    },
    'Tuberculosis': {
        'description': 'A bacterial infection that primarily affects the lungs but can spread to other organs.',
        'duration': '6-9 months of treatment',
        'risk_level': 'High',
        'recommendations': [
            'Complete full course of antibiotics',
            'Isolate during contagious period',
            'Regular medical check-ups',
            'Cover mouth when coughing',
            'Notify close contacts for testing'
        ]
    },
    'Alzheimers': {
        'description': 'A progressive neurological disorder that causes memory loss and cognitive decline.',
        'duration': 'Progressive, typically 8-10 years after diagnosis',
        'risk_level': 'High',
        'recommendations': [
            'Seek neurological evaluation',
            'Consider medications to manage symptoms',
            'Create a safe home environment',
            'Join support groups for caregivers',
            'Plan for long-term care needs'
        ]
    },
    'Pulmonary Embolism': {
        'description': 'A blockage in one of the pulmonary arteries in the lungs, usually caused by blood clots.',
        'duration': 'Recovery takes weeks with treatment',
        'risk_level': 'Critical',
        'recommendations': [
            'Seek emergency medical care immediately',
            'Receive anticoagulant therapy',
            'Monitor for complications',
            'Wear compression stockings',
            'Follow up with healthcare provider'
        ]
    }
}

def search_web_for_disease_info(disease_name):
    """Background web search for additional disease information"""
    try:
        # This is a placeholder for web search functionality
        # In a real implementation, you would use APIs like Google Custom Search, Bing, etc.
        time.sleep(1)  # Simulate search delay
        return {
            'web_info': f'Additional information about {disease_name} from web sources',
            'sources': ['Mayo Clinic', 'WebMD', 'CDC']
        }
    except Exception as e:
        print(f"Web search error: {e}")
        return None

@app.route('/')
def home():
    return render_template('index_ultra_modern.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or vectorizer is None:
        return jsonify({'error': 'Model not loaded. Please train the model first.'}), 500
    
    try:
        data = request.json
        symptoms = data.get('symptoms', [])
        
        if not symptoms:
            return jsonify({'error': 'Please select at least one symptom'}), 400
        
        symptoms_processed = [s.lower().replace(' ', '_') for s in symptoms]
        symptoms_text = ' '.join(symptoms_processed)
        
        print(f"[DEBUG] Received symptoms: {symptoms}")
        print(f"[DEBUG] Processed symptoms: {symptoms_processed}")
        print(f"[DEBUG] Symptoms text: {symptoms_text}")
        
        symptom_vector = vectorizer.transform([symptoms_text])
        
        prediction = str(model.predict(symptom_vector)[0])
        probabilities = model.predict_proba(symptom_vector)[0]
        confidence = float(np.max(probabilities) * 100)
        
        print(f"[DEBUG] Prediction: {prediction}, Confidence: {confidence:.2f}%")
        
        is_serious = prediction in SERIOUS_CONDITIONS
        severity_message = (
            "WARNING: This may be a serious condition. Please consult a doctor immediately!" 
            if is_serious 
            else "INFO: This is a general prediction. Rest, stay hydrated, and consult a doctor if symptoms persist."
        )
        
        # Get disease information
        disease_info = DISEASE_INFO.get(prediction, {
            'description': f'Information about {prediction} is being researched by our AI system.',
            'duration': 'Duration information being analyzed...',
            'risk_level': 'Risk assessment in progress...',
            'recommendations': [
                'Consult a healthcare professional for accurate diagnosis',
                'Monitor your symptoms closely',
                'Rest and stay hydrated',
                'Seek medical attention if symptoms worsen'
            ]
        })
        
        # Start background web search (optional enhancement)
        web_search_thread = threading.Thread(
            target=search_web_for_disease_info, 
            args=(prediction,),
            daemon=True
        )
        web_search_thread.start()
        
        # Get top 3 predictions with confidence
        top_indices = np.argsort(probabilities)[-3:][::-1]
        top_predictions = []
        for idx in top_indices:
            disease_name = model.classes_[idx]
            conf = float(probabilities[idx] * 100)
            top_predictions.append({
                'disease': disease_name,
                'confidence': f"{conf:.1f}%"
            })
        
        response_data = {
            'predicted_disease': prediction,
            'confidence': f"{confidence:.1f}%",
            'severity': 'Critical' if is_serious and prediction in ['Heart Attack', 'Stroke', 'Meningitis', 'Pulmonary Embolism'] else ('Serious' if is_serious else 'Moderate'),
            'message': severity_message,
            'disclaimer': 'This system is for informational purposes only and not a substitute for professional medical advice.',
            'disease_info': disease_info,
            'top_predictions': top_predictions
        }
        print(f"[DEBUG] Response: {response_data}")
        return jsonify(response_data)
        
    except Exception as e:
        error_msg = str(e)
        print(f"[ERROR] {error_msg}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': error_msg}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)