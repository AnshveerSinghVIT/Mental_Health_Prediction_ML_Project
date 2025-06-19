#We Rise Again
from flask import Flask, render_template, request, session, redirect, url_for
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
import random
import time

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

try:
    model = joblib.load('model.pkl')
    ct = joblib.load('feature_values.joblib')
except Exception as e:
    print(f"Error loading model files: {e}")

DEFAULT_VALUES = {
    'Age': 30,
    'Gender': 'Male',
    'self_employed': 'No',
    'family_history': 'No',
    'work_interfere': 'Sometimes',
    'no_employees': '6-25',
    'remote_work': 'No',
    'tech_company': 'Yes',
    'benefits': 'Yes',
    'care_options': 'No',
    'wellness_program': 'No',
    'seek_help': 'Yes',
    'anonymity': 'Yes',
    'leave': 'Somewhat easy',
    'mental_health_consequence': 'No',
    'phys_health_consequence': 'No',
    'coworkers': 'Some of them',
    'supervisor': 'Yes',
    'mental_health_interview': 'No',
    'phys_health_interview': 'No',
    'mental_vs_physical': 'Yes',
    'obs_consequence': 'No'
}


EMPLOYER_SPECIFIC_QUESTIONS = [
    'no_employees', 'tech_company', 'benefits', 'care_options',
    'wellness_program', 'seek_help', 'anonymity', 'leave',
    'mental_health_consequence', 'phys_health_consequence',
    'coworkers', 'supervisor', 'mental_vs_physical', 'obs_consequence'
]

RESPONSE_MAPPING = {
    'leave': {
        'Very easy': 'Very easy',
        'Somewhat easy': 'Somewhat easy',
        'Somewhat difficult': 'Somewhat difficult',
        'Very difficult': 'Very difficult',
        'I don\'t know': 'Don\'t know',
        'easy': 'Somewhat easy',
        'difficult': 'Somewhat difficult'
    },
    'work_interfere': {
        'I don\'t experience mental health issues': 'Never'
    },
    'phys_health_consequence': {
        'Yes': 'Yes',
        'No': 'No',
        'Maybe': 'Maybe',
        'them': 'Maybe' 
    },
    'mental_health_consequence': {
        'Yes': 'Yes',
        'No': 'No',
        'Maybe': 'Maybe'
    },
    'mental_health_interview': {
        'Yes': 'Yes',
        'No': 'No',
        'Maybe': 'Maybe'
    },
    'phys_health_interview': {
        'Yes': 'Yes',
        'No': 'No',
        'Maybe': 'Maybe'
    }
}

QUESTIONS = [
    {
        "id": "Age",
        "text": "Let's start with some basic information. How old are you?",
        "type": "number",
        "min": 18,
        "max": 100,
        "required": True,
        "emoji": "🎂",
        "category": "Demographics"
    },
    {
        "id": "Gender",
        "text": "How do you identify your gender?",
        "type": "select",
        "options": ["Male", "Female", "Non-Binary", "Transgender", "Other", "Prefer not to say"],
        "required": True,
        "emoji": "🚻",
        "category": "Demographics"
    },
    {
        "id": "self_employed",
        "text": "Are you self-employed?",
        "type": "select",
        "options": ["Yes", "No"],
        "required": True,
        "emoji": "💼",
        "category": "Employment"
    },
    {
        "id": "family_history",
        "text": "Has anyone in your immediate family been diagnosed with a mental health condition?",
        "type": "select",
        "options": ["Yes", "No", "Not sure"],
        "required": True,
        "emoji": "👨‍👩‍👧‍👦",
        "category": "Family History",
        "sensitive": True
    },
    {
        "id": "work_interfere",
        "text": "If you've experienced mental health issues, how often do they interfere with your work?",
        "type": "select",
        "options": ["Never", "Rarely", "Sometimes", "Often", "I don't experience mental health issues"],
        "required": True,
        "emoji": "📉",
        "category": "Work Impact",
        "sensitive": True
    },
    {
        "id": "no_employees",
        "text": "How many people work at your company?",
        "type": "select",
        "options": ["1-5", "6-25", "26-100", "100-500", "500+"],
        "required": True,
        "emoji": "🏢",
        "category": "Employment",
        "condition": lambda answers: answers.get('self_employed') == 'No'
    },
    {
        "id": "remote_work",
        "text": "Do you work remotely (outside of an office) at least 50% of the time?",
        "type": "select",
        "options": ["Yes", "No"],
        "required": True,
        "emoji": "🏠",
        "category": "Work Environment"
    },
    {
        "id": "tech_company",
        "text": "Is your employer primarily a tech company/organization?",
        "type": "select",
        "options": ["Yes", "No", "Not sure"],
        "required": True,
        "emoji": "💻",
        "category": "Employment",
        "condition": lambda answers: answers.get('self_employed') == 'No'
    },
    {
        "id": "benefits",
        "text": "Does your employer provide mental health benefits as part of healthcare coverage?",
        "type": "select",
        "options": ["Yes", "No", "Don't know"],
        "required": True,
        "emoji": "🏥",
        "category": "Work Benefits",
        "condition": lambda answers: answers.get('self_employed') == 'No'
    },
    {
        "id": "care_options",
        "text": "Do you know the options for mental health care available under your employer-provided coverage?",
        "type": "select",
        "options": ["Yes", "No", "Not sure"],
        "required": True,
        "emoji": "ℹ",
        "category": "Work Benefits",
        "condition": lambda answers: answers.get('self_employed') == 'No'
    },
    {
        "id": "wellness_program",
        "text": "Has your employer ever discussed mental health as part of an employee wellness program?",
        "type": "select",
        "options": ["Yes", "No", "Don't know"],
        "required": True,
        "emoji": "💬",
        "category": "Work Environment",
        "condition": lambda answers: answers.get('self_employed') == 'No'
    },
    {
        "id": "seek_help",
        "text": "Does your employer provide resources to learn more about mental health issues and how to seek help?",
        "type": "select",
        "options": ["Yes", "No", "Don't know"],
        "required": True,
        "emoji": "🆘",
        "category": "Work Benefits",
        "condition": lambda answers: answers.get('self_employed') == 'No'
    },
    {
        "id": "anonymity",
        "text": "Is your anonymity protected if you choose to take advantage of mental health treatment programs?",
        "type": "select",
        "options": ["Yes", "No", "Don't know"],
        "required": True,
        "emoji": "🕵",
        "category": "Work Benefits",
        "sensitive": True,
        "condition": lambda answers: answers.get('self_employed') == 'No'
    },
    {
        "id": "leave",
        "text": "How easy is it for you to take medical leave for a mental health condition?",
        "type": "select",
        "options": ["Very easy", "Somewhat easy", "Somewhat difficult", "Very difficult", "I don't know"],
        "required": True,
        "emoji": "⏸",
        "category": "Work Environment",
        "sensitive": True,
        "condition": lambda answers: answers.get('self_employed') == 'No'
    },
    {
        "id": "mental_health_consequence",
        "text": "Do you think that discussing a mental health issue with your employer would have negative consequences?",
        "type": "select",
        "options": ["Yes", "No", "Maybe"],
        "required": True,
        "emoji": "⚠",
        "category": "Work Environment",
        "sensitive": True,
        "condition": lambda answers: answers.get('self_employed') == 'No'
    },
    {
        "id": "phys_health_consequence",
        "text": "Do you think that discussing a physical health issue with your employer would have negative consequences?",
        "type": "select",
        "options": ["Yes", "No", "Maybe"],
        "required": True,
        "emoji": "⚠",
        "category": "Work Environment",
        "condition": lambda answers: answers.get('self_employed') == 'No'
    },
    {
        "id": "coworkers",
        "text": "Would you be willing to discuss a mental health issue with your coworkers?",
        "type": "select",
        "options": ["Yes", "No", "Some of them"],
        "required": True,
        "emoji": "👥",
        "category": "Work Relationships",
        "sensitive": True,
        "condition": lambda answers: answers.get('self_employed') == 'No'
    },
    {
        "id": "supervisor",
        "text": "Would you be willing to discuss a mental health issue with your direct supervisor(s)?",
        "type": "select",
        "options": ["Yes", "No"],
        "required": True,
        "emoji": "👔",
        "category": "Work Relationships",
        "sensitive": True,
        "condition": lambda answers: answers.get('self_employed') == 'No'
    },
    {
        "id": "mental_health_interview",
        "text": "Would you bring up a mental health issue with a potential employer in an interview?",
        "type": "select",
        "options": ["Yes", "No", "Maybe"],
        "required": True,
        "emoji": "💼",
        "category": "Work Relationships",
        "sensitive": True
    },
    {
        "id": "phys_health_interview",
        "text": "Would you bring up a physical health issue with a potential employer in an interview?",
        "type": "select",
        "options": ["Yes", "No", "Maybe"],
        "required": True,
        "emoji": "💪",
        "category": "Work Relationships"
    },
    {
        "id": "mental_vs_physical",
        "text": "Do you feel that your employer takes mental health as seriously as physical health?",
        "type": "select",
        "options": ["Yes", "No", "Don't know"],
        "required": True,
        "emoji": "⚖",
        "category": "Work Environment",
        "condition": lambda answers: answers.get('self_employed') == 'No'
    },
    {
        "id": "obs_consequence",
        "text": "Have you heard of or observed negative consequences for coworkers with mental health conditions in your workplace?",
        "type": "select",
        "options": ["Yes", "No"],
        "required": True,
        "emoji": "👀",
        "category": "Work Environment",
        "sensitive": True,
        "condition": lambda answers: answers.get('self_employed') == 'No'
    }
]

def get_next_question_index(current_index, answers):
    """Determine the next question index, skipping questions that don't apply"""
    next_index = current_index + 1
    
    while next_index < len(QUESTIONS):
        question = QUESTIONS[next_index]
   
        if 'condition' in question and not question['condition'](answers):
            next_index += 1
        else:
            break
    
    return next_index

@app.route('/')
def home():
    session.clear()
    return render_template('home.html')

@app.route('/assessment/start')
def start_assessment():
    session['question_index'] = 0
    session['answers'] = {}
    return redirect(url_for('show_question'))

@app.route('/assessment/question')
def show_question():
    if 'question_index' not in session:
        return redirect(url_for('start_assessment'))
    
    question_index = session['question_index']
    if question_index >= len(QUESTIONS):
        return redirect(url_for('complete_assessment'))
    
   
    question = QUESTIONS[question_index]
    
   
    if 'condition' in question and not question['condition'](session.get('answers', {})):
        session['question_index'] = get_next_question_index(question_index, session.get('answers', {}))
        return redirect(url_for('show_question'))
    
   
    if question_index > 0:
        prev_answer = session['answers'].get(QUESTIONS[question_index-1]['id'], '')
        if prev_answer and random.random() > 0.7: 
            question = question.copy()
            references = [
                f"Thanks for sharing that. {question['text']}",
                f"I appreciate your honesty. {question['text']}",
                f"Understood. {question['text']}",
                f"That's helpful to know. {question['text']}"
            ]
            question['text'] = random.choice(references)
    
  
    total_questions = len([q for q in QUESTIONS if not ('condition' in q and not q['condition'](session.get('answers', {})))])
    answered_questions = len([q for i, q in enumerate(QUESTIONS[:question_index]) 
                            if not ('condition' in q and not q['condition'](session.get('answers', {})))])
    
    return render_template('question.html', 
                         question=question, 
                         progress=100*answered_questions/total_questions,
                         total_questions=total_questions,
                         current_question=answered_questions+1)

@app.route('/assessment/answer', methods=['POST'])
def process_answer():
    question_index = session['question_index']
    question = QUESTIONS[question_index]
    
    answer = request.form.get('answer')
    session['answers'][question['id']] = answer
    session.modified = True
    
   
    session['question_index'] = get_next_question_index(question_index, session['answers'])
    return redirect(url_for('show_question'))

def standardize_response(question_id, response):
    """Standardize responses to match what the model expects"""
    if not response or str(response).strip() == '':
        return DEFAULT_VALUES.get(question_id, 'Unknown')
    
    response = str(response).strip()
    
    if question_id in RESPONSE_MAPPING:
        mapping = RESPONSE_MAPPING[question_id]
        if response in mapping:
            return mapping[response]
        for key in mapping:
            if key.lower() in response.lower():
                return mapping[key]
    
    if question_id == 'Gender':
        gender_map = {
            'male': 'Male',
            'female': 'Female',
            'non-binary': 'Non-Binary',
            'trans': 'Transgender',
            'other': 'Other',
            'prefer not to say': 'Prefer not to say'
        }
        for key in gender_map:
            if key in response.lower():
                return gender_map[key]
        return 'Other' 
    
    question = next((q for q in QUESTIONS if q['id'] == question_id), None)
    if question:
        return response if response in question.get('options', []) else DEFAULT_VALUES.get(question_id, 'Unknown')
    
    return DEFAULT_VALUES.get(question_id, 'Unknown')

@app.route('/assessment/complete')
def complete_assessment():
   
    data = DEFAULT_VALUES.copy()
    
   
    for key, value in session.get('answers', {}).items():
        if isinstance(value, str) and ' ' in value:
            value = value.split(' ')[-1] 
        
        standardized_value = standardize_response(key, value)
        
        if key == 'Age':  
            try:
                data[key] = int(standardized_value) if standardized_value else DEFAULT_VALUES[key]
                data[key] = max(18, min(100, data[key]))
            except (ValueError, TypeError):
                data[key] = DEFAULT_VALUES[key]
        else: 
            data[key] = str(standardized_value).strip() if standardized_value else DEFAULT_VALUES[key]
    
   
    if data.get('self_employed') == 'Yes':
        for question in EMPLOYER_SPECIFIC_QUESTIONS:
            if question not in data:
                if question in ['no_employees', 'tech_company']:
                    data[question] = 'N/A (Self-employed)'
                else:
                    data[question] = DEFAULT_VALUES.get(question, 'Unknown')
    
  
    input_df = pd.DataFrame([data])
    
    try:
       
        expected_columns = ct.feature_names_in_
        
        for col in expected_columns:
            if col not in input_df.columns:
                input_df[col] = DEFAULT_VALUES.get(col, 'Unknown')
        
        input_df = input_df[expected_columns]
        
        app.logger.info(f"Input data for model: {input_df.to_dict('records')[0]}")
        
      
        X = ct.transform(input_df)
        prediction = model.predict(X)[0]
        
       
        time.sleep(2)
        
        return render_template('output.html', prediction=prediction)
    
    except Exception as e:
        app.logger.error(f"Error during transformation: {str(e)}")
        return render_template('error.html', 
                            error="We encountered an issue processing your responses. Please try again.")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9090, debug=True)