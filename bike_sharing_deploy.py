from flask import Flask, request, jsonify 
import joblib 

# inisiasi aplikasi Flask 
app = Flask(__name__)

# creating the model that is saved
joblib_model = joblib.load('rfrmodel_bike_sharing.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['bikesharing_testdata'] #taking data from request json 
    prediction = joblib_model.predict(data) #predict in 2D array 
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True) 

