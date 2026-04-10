from flask import Flask, render_template, request, jsonify
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import pickle
import os
import traceback

app = Flask(__name__)

# Get absolute path to the app directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load and prepare data
try:
    df = pd.read_csv(os.path.join(BASE_DIR, 'vgsales.csv'))
    df = df.dropna(subset=['Critic_Score', 'User_Score', 'Global_Sales', 'Genre', 'Platform'])
    
    # Encode categorical variables
    le_genre = LabelEncoder()
    le_platform = LabelEncoder()
    df['Genre_encoded'] = le_genre.fit_transform(df['Genre'])
    df['Platform_encoded'] = le_platform.fit_transform(df['Platform'])
    
    # Features and target
    X = df[['Genre_encoded', 'Platform_encoded', 'Critic_Score']]
    y = df['Global_Sales']
    
    # Train model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Save encoders and model with absolute paths
    with open(os.path.join(BASE_DIR, 'le_genre.pkl'), 'wb') as f:
        pickle.dump(le_genre, f)
    with open(os.path.join(BASE_DIR, 'le_platform.pkl'), 'wb') as f:
        pickle.dump(le_platform, f)
    with open(os.path.join(BASE_DIR, 'model.pkl'), 'wb') as f:
        pickle.dump(model, f)
    
    print("Model trained and saved successfully.")
except Exception as e:
    print(f"Error during model training: {e}")
    traceback.print_exc()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/Generos/')
def generos():
    # Data for genres
    genre_sales = df.groupby('Genre')[['NA_Sales', 'EU_Sales', 'JP_Sales', 'Global_Sales']].sum().reset_index()
    genres = genre_sales['Genre'].tolist()
    na_sales = genre_sales['NA_Sales'].tolist()
    eu_sales = genre_sales['EU_Sales'].tolist()
    jp_sales = genre_sales['JP_Sales'].tolist()
    
    # Top 10 games by region
    top_na = df.nlargest(10, 'NA_Sales')[['Name', 'NA_Sales']].to_dict('records')
    top_eu = df.nlargest(10, 'EU_Sales')[['Name', 'EU_Sales']].to_dict('records')
    top_jp = df.nlargest(10, 'JP_Sales')[['Name', 'JP_Sales']].to_dict('records')
    
    return render_template('Generos.html', 
                           genres=json.dumps(genres), 
                           na_sales=json.dumps(na_sales), 
                           eu_sales=json.dumps(eu_sales), 
                           jp_sales=json.dumps(jp_sales),
                           top_na=json.dumps(top_na),
                           top_eu=json.dumps(top_eu),
                           top_jp=json.dumps(top_jp))

@app.route('/criticos')
def criticos():
    critics_data = {
        'x': df['Critic_Score'].tolist(),
        'y': df['Global_Sales'].tolist(),
        'text': df['Name'].tolist()
    }
    users_data = {
        'x': (df['User_Score'] * 10).tolist(),  # Assuming User_Score is out of 10
        'y': df['Global_Sales'].tolist(),
        'text': df['Name'].tolist()
    }
    
    return render_template('Criticos.html', critics_data=json.dumps(critics_data), users_data=json.dumps(users_data))

@app.route('/predictor')
def predictor():
    return render_template('predictor.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        genre = request.form.get('genre', '').strip()
        platform = request.form.get('platform', '').strip()
        rating_str = request.form.get('rating', '').strip()
        
        # Validate inputs
        if not genre or not platform or not rating_str:
            prediction = "Error: Por favor completa todos los campos."
            return render_template('predictor.html', prediction=prediction)
        
        try:
            rating = float(rating_str)
            if rating < 0 or rating > 100:
                raise ValueError("La calificación debe estar entre 0 y 100")
        except ValueError as e:
            prediction = f"Error: {str(e)}"
            return render_template('predictor.html', prediction=prediction)
        
        # Load model and encoders with absolute paths
        try:
            with open(os.path.join(BASE_DIR, 'model.pkl'), 'rb') as f:
                model = pickle.load(f)
            with open(os.path.join(BASE_DIR, 'le_genre.pkl'), 'rb') as f:
                le_genre = pickle.load(f)
            with open(os.path.join(BASE_DIR, 'le_platform.pkl'), 'rb') as f:
                le_platform = pickle.load(f)
        except FileNotFoundError:
            prediction = "Error: El modelo no ha sido entrenado. Por favor reinicia la aplicación."
            return render_template('predictor.html', prediction=prediction)
        
        # Encode inputs
        try:
            genre_encoded = le_genre.transform([genre])[0]
            platform_encoded = le_platform.transform([platform])[0]
        except ValueError as e:
            valid_genres = ', '.join(le_genre.classes_)
            valid_platforms = ', '.join(le_platform.classes_)
            prediction = f"Error: Género o plataforma no válida.\nGéneros válidos: {valid_genres}\nPlataformas válidas: {valid_platforms}"
            return render_template('predictor.html', prediction=prediction)
        
        # Normalize rating to 0-100 scale
        normalized_rating = min(100, max(0, rating))
        
        # Predict
        prediction_value = model.predict([[genre_encoded, platform_encoded, normalized_rating]])[0]
        
        if prediction_value > 15:
            success = "Éxito muy alto"
        elif prediction_value > 10:
            success = "Éxito alto"
        elif prediction_value > 5:
            success = "Éxito moderado"
        elif prediction_value > 2:
            success = "Éxito bajo"
        else:
            success = "Éxito muy bajo"
        
        prediction = f"✓ Predicción de ventas globales: <strong>{prediction_value:.2f} millones</strong><br>Nivel de éxito: <strong>{success}</strong>"
        return render_template('predictor.html', prediction=prediction)
    
    except Exception as e:
        print(f"Error in predict: {e}")
        traceback.print_exc()
        prediction = f"Error inesperado: {str(e)}"
        return render_template('predictor.html', prediction=prediction)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions - returns JSON"""
    try:
        genre = request.form.get('genre', '').strip()
        platform = request.form.get('platform', '').strip()
        rating_str = request.form.get('rating', '').strip()
        
        print(f"[DEBUG] Predict request: genre={genre}, platform={platform}, rating={rating_str}")
        
        # Validate inputs
        if not genre or not platform or not rating_str:
            print("[DEBUG] Validation failed: missing fields")
            return jsonify({'success': False, 'error': 'Por favor completa todos los campos.'}), 400
        
        try:
            rating = float(rating_str)
            if rating < 0 or rating > 100:
                raise ValueError("La calificación debe estar entre 0 y 100")
        except ValueError as e:
            print(f"[DEBUG] Rating validation failed: {str(e)}")
            return jsonify({'success': False, 'error': str(e)}), 400
        
        # Load model and encoders
        try:
            print(f"[DEBUG] Loading model from: {os.path.join(BASE_DIR, 'model.pkl')}")
            with open(os.path.join(BASE_DIR, 'model.pkl'), 'rb') as f:
                model = pickle.load(f)
            with open(os.path.join(BASE_DIR, 'le_genre.pkl'), 'rb') as f:
                le_genre = pickle.load(f)
            with open(os.path.join(BASE_DIR, 'le_platform.pkl'), 'rb') as f:
                le_platform = pickle.load(f)
            print("[DEBUG] Model and encoders loaded successfully")
        except FileNotFoundError as fnf:
            print(f"[DEBUG] File not found: {fnf}")
            return jsonify({'success': False, 'error': 'El modelo no ha sido entrenado. Por favor reinicia la aplicación.'}), 500
        
        # Encode inputs
        try:
            print(f"[DEBUG] Encoding inputs: genre={genre}, platform={platform}")
            genre_encoded = le_genre.transform([genre])[0]
            platform_encoded = le_platform.transform([platform])[0]
            print(f"[DEBUG] Encoded: genre_encoded={genre_encoded}, platform_encoded={platform_encoded}")
        except ValueError as e:
            print(f"[DEBUG] Encoding failed: {str(e)}")
            valid_genres = ', '.join(le_genre.classes_)
            valid_platforms = ', '.join(le_platform.classes_)
            return jsonify({'success': False, 'error': f'Género o plataforma no válida. Géneros: {valid_genres}. Plataformas: {valid_platforms}'}), 400
        
        # Predict
        print(f"[DEBUG] Making prediction with: genre_encoded={genre_encoded}, platform_encoded={platform_encoded}, rating={rating}")
        prediction_value = model.predict([[genre_encoded, platform_encoded, rating]])[0]
        print(f"[DEBUG] Prediction result: {prediction_value}")
        
        if prediction_value > 15:
            success = "Éxito muy alto"
        elif prediction_value > 10:
            success = "Éxito alto"
        elif prediction_value > 5:
            success = "Éxito moderado"
        elif prediction_value > 2:
            success = "Éxito bajo"
        else:
            success = "Éxito muy bajo"
        
        response = {
            'success': True,
            'sales': f"{prediction_value:.2f}",
            'level': success
        }
        print(f"[DEBUG] Sending response: {response}")
        return jsonify(response)
    
    except Exception as e:
        print(f"[ERROR] Error in api_predict: {e}")
        traceback.print_exc()
        error_response = {'success': False, 'error': f'Error: {str(e)}'}
        print(f"[DEBUG] Sending error response: {error_response}")
        return jsonify(error_response), 500

if __name__ == '__main__':
    app.run(debug=True, port=5500, host='127.0.0.1')