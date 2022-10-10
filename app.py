import os
import logging

from flask import Flask, render_template_string, request, jsonify, render_template
from model import translation

app = Flask(__name__, template_folder='template')  

# define model path
model_path = 'model1.h5'

# create instance
model = translation(model_path)
logging.basicConfig(level=logging.INFO)

@app.route("/")
def index():
    """Provide simple health check route."""
    return render_template('Translation_template.html')

@app.route("/", methods=['GET', 'POST'])
def predict():
    """Provide main prediction API route. Responds to both GET and POST requests."""
    logging.info("Predict request received!")
    english_sentence = request.form["sentence"]
    french_sentence = model.predict(english_sentence)
    
    logging.info("prediction from model= {}".format(french_sentence))
    return render_template('Translation_template.html', translation = french_sentence)

def main():
    """Run the Flask app."""
    port=int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port = port, debug=True) 

if __name__ == "__main__":
    main()