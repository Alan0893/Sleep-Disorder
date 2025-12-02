from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Global variables to store models and data
df = None
scaler = None
km4 = None
cluster_profile = None
cluster_risk = None
cluster_counts = None

def load_model_and_data():
    """Load the data and train the clustering model"""
    global df, scaler, km4, cluster_profile, cluster_risk, cluster_counts
    
    # Load data
    df = pd.read_csv('/Users/alanlin/Desktop/BA 305/Project/Sleep_health_and_lifestyle_dataset.csv')
    
    # Preprocess
    df['Sleep Disorder'].fillna('None', inplace=True)
    bp_split = df['Blood Pressure'].str.split('/', expand=True)
    df['Systolic_BP'] = bp_split[0].astype(int)
    df['Diastolic_BP'] = bp_split[1].astype(int)
    
    # Binary target for risk calculation
    df['Has_Sleep_Disorder'] = ((df['Sleep Disorder'] == 'Insomnia') | 
                                (df['Sleep Disorder'] == 'Sleep Apnea')).astype(int)
    
    # Clustering features
    cluster_features = ['Age', 'Sleep Duration', 'Quality of Sleep', 'Physical Activity Level',
                        'Stress Level', 'Heart Rate', 'Daily Steps', 'Systolic_BP', 'Diastolic_BP']
    
    # Standardize and train
    scaler = StandardScaler()
    X_cluster = scaler.fit_transform(df[cluster_features])
    
    km4 = KMeans(n_clusters=4, random_state=42, n_init=10)
    df['Cluster4'] = km4.fit_predict(X_cluster)
    
    # Profile clusters
    profile_cols = cluster_features + ['Has_Sleep_Disorder']
    cluster_profile = df.groupby('Cluster4')[profile_cols].mean().round(2)
    cluster_risk = df.groupby('Cluster4')['Has_Sleep_Disorder'].mean()
    cluster_counts = df['Cluster4'].value_counts().sort_index()
    
    return df, scaler, km4, cluster_profile, cluster_risk, cluster_counts

# Load on startup
df, scaler, km4, cluster_profile, cluster_risk, cluster_counts = load_model_and_data()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Extract user inputs
        user_data = np.array([[
            float(data['age']),
            float(data['sleep_duration']),
            float(data['quality_of_sleep']),
            float(data['physical_activity_level']),
            float(data['stress_level']),
            float(data['heart_rate']),
            float(data['daily_steps']),
            float(data['systolic_bp']),
            float(data['diastolic_bp'])
        ]])
        
        # Standardize and predict
        user_scaled = scaler.transform(user_data)
        cluster = int(km4.predict(user_scaled)[0])
        
        # Get cluster statistics
        cluster_stats = cluster_profile.loc[cluster].to_dict()
        cluster_risk_pct = float(cluster_risk.loc[cluster] * 100)
        cluster_size = int(cluster_counts.loc[cluster])
        total_size = int(cluster_counts.sum())
        
        # Cluster names
        cluster_names = {
            0: "The Well-Balanced Sleeper",
            1: "The Mature Health Guardian",
            2: "The Stress-Affected Individual",
            3: "The High-Risk Professional"
        }
        
        cluster_descriptions = {
            0: "You maintain excellent sleep habits with good sleep duration and quality. Your stress levels are manageable, and your lifestyle shows a healthy balance.",
            1: "You're in an older age group with good sleep quality, but you face moderate risk factors. Your lifestyle reflects someone who may have accumulated health factors over time.",
            2: "You're experiencing challenges with sleep duration and quality, likely related to elevated stress levels. This cluster shows the impact of stress on sleep health.",
            3: "You're in a high-activity, high-stress lifestyle with compromised sleep. This cluster represents individuals with demanding lifestyles who may be sacrificing sleep for other commitments."
        }
        
        cluster_recommendations = {
            0: "Keep up the excellent work! Maintain your current sleep routine, stress management practices, and physical activity levels.",
            1: "Focus on cardiovascular health and monitor your blood pressure regularly. Consider stress reduction techniques and ensure adequate rest.",
            2: "Priority actions: Focus on stress management and sleep hygiene. Consider meditation, relaxation techniques, or counseling.",
            3: "Urgent attention needed: Re-evaluate work-life balance, implement strict sleep schedules, and consider medical consultation for sleep disorders."
        }
        
        # Prepare response
        response = {
            'cluster': cluster,
            'cluster_name': cluster_names[cluster],
            'description': cluster_descriptions[cluster],
            'recommendation': cluster_recommendations[cluster],
            'risk_percentage': cluster_risk_pct,
            'risk_level': 'LOW' if cluster_risk_pct < 20 else 'MODERATE' if cluster_risk_pct < 70 else 'HIGH',
            'cluster_size': cluster_size,
            'total_size': total_size,
            'cluster_percentage': round((cluster_size / total_size) * 100, 1),
            'user_metrics': {
                'age': float(data['age']),
                'sleep_duration': float(data['sleep_duration']),
                'quality_of_sleep': float(data['quality_of_sleep']),
                'physical_activity_level': float(data['physical_activity_level']),
                'stress_level': float(data['stress_level']),
                'heart_rate': float(data['heart_rate']),
                'daily_steps': float(data['daily_steps']),
                'systolic_bp': float(data['systolic_bp']),
                'diastolic_bp': float(data['diastolic_bp'])
            },
            'cluster_averages': {
                'age': float(cluster_stats['Age']),
                'sleep_duration': float(cluster_stats['Sleep Duration']),
                'quality_of_sleep': float(cluster_stats['Quality of Sleep']),
                'physical_activity_level': float(cluster_stats['Physical Activity Level']),
                'stress_level': float(cluster_stats['Stress Level']),
                'heart_rate': float(cluster_stats['Heart Rate']),
                'daily_steps': float(cluster_stats['Daily Steps']),
                'systolic_bp': float(cluster_stats['Systolic_BP']),
                'diastolic_bp': float(cluster_stats['Diastolic_BP'])
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/cluster_info')
def cluster_info():
    """Get information about all clusters"""
    cluster_info = {}
    cluster_names = ['Well-Balanced Sleeper', 'Mature Health Guardian', 
                    'Stress-Affected Individual', 'High-Risk Professional']
    
    for i in range(4):
        cluster_info[i] = {
            'name': cluster_names[i],
            'count': int(cluster_counts.loc[i]),
            'percentage': round((cluster_counts.loc[i] / cluster_counts.sum()) * 100, 1),
            'risk': round(float(cluster_risk.loc[i] * 100), 1)
        }
    
    return jsonify(cluster_info)

if __name__ == '__main__':
    app.run(debug=True, port=5000)

