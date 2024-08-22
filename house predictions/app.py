from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the pickled model (replace 'house_model.pkl' with your actual model file)
model = pickle.load(open('house-model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        location = request.form['location']
        total_sqft = float(request.form['total_sqft'])
        bath = int(request.form['bath'])
        bhk = int(request.form['bhk'])
        # Predict using our machine learning model (adjust this based on our model)
        features = [total_sqft, bath,bhk]
        predicted_price = model.predict([features])[0]

        #output=predicted_price[0]
        return render_template('index.html', prediction=f'Predicted Price: ${predicted_price:.2f}')
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)


