from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings
import os
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Global variables to store models and data
df = None
scaler_cluster = None
pca = None
kmeans = None
le_gender = None
le_occupation = None
le_bmi = None

def load_model_and_data():
    """Load the data and train the clustering model with PCA"""
    global df, scaler_cluster, pca, kmeans, le_gender, le_occupation, le_bmi
    
    # Load data - try multiple possible paths
    possible_paths = [
        os.path.join(os.path.dirname(__file__), '..', 'Project', 'Sleep_health_and_lifestyle_dataset.csv'),
        os.path.join(os.path.dirname(__file__), 'Sleep_health_and_lifestyle_dataset.csv'),
        os.path.join(os.path.dirname(__file__), '..', 'Sleep_health_and_lifestyle_dataset.csv')
    ]
    
    data_path = None
    for path in possible_paths:
        if os.path.exists(path):
            data_path = path
            break
    
    if data_path is None:
        raise FileNotFoundError("Could not find Sleep_health_and_lifestyle_dataset.csv. Please ensure the file exists.")
    
    df = pd.read_csv(data_path)
    
    # Preprocess
    df['Sleep Disorder'].fillna('None', inplace=True)
    df['BMI Category'] = df['BMI Category'].replace('Normal Weight', 'Normal')
    
    # Parse Blood Pressure
    bp_split = df['Blood Pressure'].str.split('/', expand=True)
    df['Systolic_BP'] = bp_split[0].astype(int)
    df['Diastolic_BP'] = bp_split[1].astype(int)
    
    # Binary target for risk calculation
    df['Has_Sleep_Disorder'] = ((df['Sleep Disorder'] == 'Insomnia') | 
                                (df['Sleep Disorder'] == 'Sleep Apnea')).astype(int)
    
    # Encode categorical variables
    le_gender = LabelEncoder()
    df['Gender_Encoded'] = le_gender.fit_transform(df['Gender'])
    
    le_occupation = LabelEncoder()
    df['Occupation_Encoded'] = le_occupation.fit_transform(df['Occupation'])
    
    le_bmi = LabelEncoder()
    df['BMI_Category_Encoded'] = le_bmi.fit_transform(df['BMI Category'])
    
    # Clustering features (same as notebook)
    numerical_features = ['Age', 'Sleep Duration', 'Quality of Sleep', 'Physical Activity Level',
                         'Stress Level', 'Heart Rate', 'Daily Steps', 'Systolic_BP', 'Diastolic_BP',
                         'Gender_Encoded', 'Occupation_Encoded', 'BMI_Category_Encoded']
    
    # Standardize and apply PCA
    scaler_cluster = StandardScaler()
    X_cluster = df[numerical_features].copy()
    X_cluster_scaled = scaler_cluster.fit_transform(X_cluster)
    
    # Apply PCA with 3 components (as per notebook)
    pca = PCA(n_components=3, random_state=42)
    X_pca = pca.fit_transform(X_cluster_scaled)
    
    # Fit KMeans with 4 clusters
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10, max_iter=300)
    df['Lifestyle_Health_Profile'] = kmeans.fit_predict(X_pca)
    
    return df, scaler_cluster, pca, kmeans, le_gender, le_occupation, le_bmi

# Load on startup
df, scaler_cluster, pca, kmeans, le_gender, le_occupation, le_bmi = load_model_and_data()

@app.route('/')
def index():
    return render_template('recommendation.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.json
        
        # Extract user inputs
        user_age = float(data['age'])
        user_gender = data['gender']
        user_occupation = data['occupation']
        user_sleep_duration = float(data['sleep_duration'])
        user_sleep_quality = float(data['quality_of_sleep'])
        user_physical_activity = float(data['physical_activity_level'])
        user_stress_level = float(data['stress_level'])
        user_daily_steps = float(data['daily_steps'])
        user_blood_pressure = data['blood_pressure']  # Format: "120/80"
        user_heart_rate = float(data['heart_rate'])
        user_bmi_category = data['bmi_category']
        
        # Parse blood pressure
        systolic_bp, diastolic_bp = map(int, user_blood_pressure.split('/'))
        
        # Encode categorical variables (handle missing values)
        try:
            gender_encoded = le_gender.transform([user_gender])[0]
        except ValueError:
            gender_encoded = le_gender.transform([df['Gender'].mode()[0]])[0]
        
        try:
            occupation_encoded = le_occupation.transform([user_occupation])[0]
        except ValueError:
            occupation_encoded = le_occupation.transform([df['Occupation'].mode()[0]])[0]
        
        try:
            bmi_encoded = le_bmi.transform([user_bmi_category])[0]
        except ValueError:
            bmi_encoded = le_bmi.transform([df['BMI Category'].mode()[0]])[0]
        
        # Create feature array in the same order as numerical_features
        numerical_features = ['Age', 'Sleep Duration', 'Quality of Sleep', 'Physical Activity Level',
                             'Stress Level', 'Heart Rate', 'Daily Steps', 'Systolic_BP', 'Diastolic_BP',
                             'Gender_Encoded', 'Occupation_Encoded', 'BMI_Category_Encoded']
        
        user_features = np.array([[
            user_age,
            user_sleep_duration,
            user_sleep_quality,
            user_physical_activity,
            user_stress_level,
            user_heart_rate,
            user_daily_steps,
            systolic_bp,
            diastolic_bp,
            gender_encoded,
            occupation_encoded,
            bmi_encoded
        ]])
        
        # Scale and apply PCA
        user_features_scaled = scaler_cluster.transform(user_features)
        user_features_pca = pca.transform(user_features_scaled)
        
        # Predict cluster
        predicted_cluster = kmeans.predict(user_features_pca)[0]
        
        # Get cluster statistics
        cluster_data = df[df['Lifestyle_Health_Profile'] == predicted_cluster]
        cluster_size = len(cluster_data)
        cluster_pct = (cluster_size / len(df)) * 100
        sleep_disorder_rate = cluster_data['Has_Sleep_Disorder'].mean() * 100
        
        # Calculate cluster averages
        cluster_avg_age = float(cluster_data['Age'].mean())
        cluster_avg_sleep_duration = float(cluster_data['Sleep Duration'].mean())
        cluster_avg_sleep_quality = float(cluster_data['Quality of Sleep'].mean())
        cluster_avg_stress = float(cluster_data['Stress Level'].mean())
        cluster_avg_physical_activity = float(cluster_data['Physical Activity Level'].mean())
        cluster_avg_daily_steps = float(cluster_data['Daily Steps'].mean())
        cluster_avg_heart_rate = float(cluster_data['Heart Rate'].mean())
        cluster_avg_systolic_bp = float(cluster_data['Systolic_BP'].mean())
        cluster_avg_diastolic_bp = float(cluster_data['Diastolic_BP'].mean())
        
        # Get recommendations based on cluster
        recommendations = get_recommendations(predicted_cluster, user_sleep_duration, 
                                             user_sleep_quality, user_stress_level,
                                             cluster_avg_sleep_duration, cluster_avg_sleep_quality,
                                             cluster_avg_stress)
        
        # Cluster names and descriptions
        cluster_info = get_cluster_info(predicted_cluster)
        
        # Prepare response
        response = {
            'cluster': int(predicted_cluster),
            'cluster_name': cluster_info['name'],
            'description': cluster_info['description'],
            'recommendations': recommendations,
            'risk_percentage': round(float(sleep_disorder_rate), 1),
            'risk_level': 'LOW' if sleep_disorder_rate < 20 else 'MODERATE' if sleep_disorder_rate < 70 else 'HIGH',
            'cluster_size': cluster_size,
            'total_size': len(df),
            'cluster_percentage': round(cluster_pct, 1),
            'user_metrics': {
                'age': user_age,
                'gender': user_gender,
                'occupation': user_occupation,
                'sleep_duration': user_sleep_duration,
                'quality_of_sleep': user_sleep_quality,
                'physical_activity_level': user_physical_activity,
                'stress_level': user_stress_level,
                'daily_steps': user_daily_steps,
                'heart_rate': user_heart_rate,
                'systolic_bp': systolic_bp,
                'diastolic_bp': diastolic_bp,
                'bmi_category': user_bmi_category
            },
            'cluster_averages': {
                'age': cluster_avg_age,
                'sleep_duration': cluster_avg_sleep_duration,
                'quality_of_sleep': cluster_avg_sleep_quality,
                'physical_activity_level': cluster_avg_physical_activity,
                'stress_level': cluster_avg_stress,
                'daily_steps': cluster_avg_daily_steps,
                'heart_rate': cluster_avg_heart_rate,
                'systolic_bp': cluster_avg_systolic_bp,
                'diastolic_bp': cluster_avg_diastolic_bp
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 400

def get_cluster_info(cluster):
    """Get cluster name and description"""
    cluster_info = {
        0: {
            'name': 'Healthy Active Professionals',
            'description': 'Low sleep disorder risk, good sleep quality, moderate stress. You maintain excellent sleep habits with good sleep duration and quality. Your stress levels are manageable, and your lifestyle shows a healthy balance.'
        },
        1: {
            'name': 'High-Risk Stressed Group',
            'description': 'HIGHEST sleep disorder risk, very high stress, poor sleep quality. This cluster represents individuals experiencing significant stress-related sleep challenges that require immediate attention.'
        },
        2: {
            'name': 'Moderate Risk Older Adults',
            'description': 'Moderate sleep disorder risk, older age, good sleep quality, low stress. You\'re in an older age group with good sleep quality, but you face moderate risk factors.'
        },
        3: {
            'name': 'Younger At-Risk Group',
            'description': 'Moderate sleep disorder risk, younger age, poor sleep duration, moderate stress. This cluster represents younger individuals who may be sacrificing sleep for other commitments.'
        }
    }
    return cluster_info.get(cluster, {'name': 'Unknown', 'description': 'Unknown cluster'})

def get_recommendations(cluster, user_sleep_dur, user_sleep_qual, user_stress,
                       avg_sleep_dur, avg_sleep_qual, avg_stress):
    """Get personalized recommendations based on cluster"""
    
    recommendations = []
    
    if cluster == 0:
        recommendations = [
            {
                'category': 'Maintain Current Habits',
                'items': [
                    'Continue your excellent sleep hygiene and active lifestyle',
                    'Keep your 7.5+ hour sleep duration consistent',
                    'Maintain your current stress management practices'
                ]
            },
            {
                'category': 'Preventive Stress Management',
                'items': [
                    'Implement stress management techniques before stress increases',
                    'Set work-life boundaries to protect your good sleep patterns',
                    'Schedule regular relaxation activities'
                ]
            },
            {
                'category': 'Regular Health Monitoring',
                'items': [
                    'Schedule annual health checkups',
                    'Maintain your normal BMI and blood pressure',
                    'Continue monitoring sleep quality'
                ]
            }
        ]
    
    elif cluster == 1:
        recommendations = [
            {
                'category': 'Immediate Medical Consultation',
                'items': [
                    'Seek sleep specialist evaluation for potential sleep apnea or insomnia',
                    'Get comprehensive health assessment',
                    'Consider sleep study if recommended by physician'
                ]
            },
            {
                'category': 'Stress Reduction Protocol',
                'items': [
                    'Consider therapy or counseling for stress management',
                    'Practice daily meditation or mindfulness (10-15 minutes)',
                    'Implement deep breathing exercises during work shifts',
                    'Establish work-life boundaries'
                ]
            },
            {
                'category': 'Sleep Hygiene Overhaul',
                'items': [
                    'Establish strict sleep schedule despite shift work',
                    'Create optimal sleep environment (blackout curtains, white noise)',
                    'Avoid caffeine 6+ hours before sleep',
                    'Limit screen time 1 hour before bed'
                ]
            },
            {
                'category': 'Physical Health Management',
                'items': [
                    'Address overweight BMI through structured diet plan',
                    'Monitor and manage elevated blood pressure',
                    'Consider reducing daily steps intensity if causing additional stress',
                    'Incorporate moderate exercise routine'
                ]
            }
        ]
    
    elif cluster == 2:
        recommendations = [
            {
                'category': 'Age-Related Sleep Adjustments',
                'items': [
                    'Maintain consistent bedtime routine despite aging',
                    'Consider earlier sleep schedule to optimize duration',
                    'Address age-related sleep architecture changes with sleep specialist',
                    'Ensure adequate sleep duration (7-9 hours)'
                ]
            },
            {
                'category': 'Cardiovascular Health Focus',
                'items': [
                    'Monitor blood pressure regularly (aim for <120/80)',
                    'Increase physical activity from current moderate levels',
                    'Aim for 7,000+ daily steps consistently',
                    'Include cardiovascular exercises in routine'
                ]
            },
            {
                'category': 'Weight Management',
                'items': [
                    'Address overweight BMI through balanced nutrition',
                    'Incorporate strength training to maintain muscle mass',
                    'Consult with nutritionist for age-appropriate diet plan'
                ]
            }
        ]
    
    elif cluster == 3:
        recommendations = [
            {
                'category': 'Sleep Duration Priority (CRITICAL)',
                'items': [
                    'Increase sleep from current level to 7-9 hours - this is critical',
                    'Set non-negotiable bedtime and wake time',
                    'Eliminate screen time 1 hour before bed',
                    'Prioritize sleep over other activities'
                ]
            },
            {
                'category': 'Stress Management for Young Professionals',
                'items': [
                    'Develop healthy coping strategies early in career',
                    'Practice time management to reduce work-related stress',
                    'Consider stress management workshops or apps',
                    'Build strong social support networks'
                ]
            },
            {
                'category': 'Lifestyle Modifications',
                'items': [
                    'Address overweight BMI before it worsens with age',
                    'Increase physical activity beyond current levels',
                    'Improve sleep quality through better sleep environment',
                    'Establish consistent daily routine'
                ]
            },
            {
                'category': 'Career-Life Balance',
                'items': [
                    'Set boundaries between work and personal time',
                    'Avoid bringing work stress into bedroom',
                    'Consider career counseling if job stress is chronic',
                    'Schedule regular breaks and downtime'
                ]
            }
        ]
    
    # Add universal recommendations
    recommendations.append({
        'category': 'Universal Prevention Strategies',
        'items': [
            'Maintain consistent sleep-wake schedule',
            'Create cool, dark, quiet sleep environment',
            'Limit caffeine and alcohol consumption',
            'Aim for 150+ minutes moderate exercise weekly',
            'Build strong social support networks',
            'Practice regular relaxation methods'
        ]
    })
    
    return recommendations

@app.route('/cluster_info')
def cluster_info():
    """Get information about all clusters"""
    cluster_info_data = {}
    cluster_names = ['Healthy Active Professionals', 'High-Risk Stressed Group', 
                    'Moderate Risk Older Adults', 'Younger At-Risk Group']
    
    for i in range(4):
        cluster_data = df[df['Lifestyle_Health_Profile'] == i]
        cluster_info_data[i] = {
            'name': cluster_names[i],
            'count': int(len(cluster_data)),
            'percentage': round((len(cluster_data) / len(df)) * 100, 1),
            'risk': round(float(cluster_data['Has_Sleep_Disorder'].mean() * 100), 1)
        }
    
    return jsonify(cluster_info_data)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(debug=True, host='0.0.0.0', port=port)

