from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('loan_model.pkl', 'rb'))

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction_text = ""
    form_visible = True  
    if request.method == 'POST':
        try:
            
            input_features = [float(x) for x in request.form.values()]
            features_value = [np.array(input_features)]
            
            prediction = model.predict(features_value)
            prediction_text = "Loan Approved" if prediction[0] == 1 else "Loan Not Approved"
            form_visible = False  
        except Exception as e:
            prediction_text = str(e)
            form_visible = True
    return render_template('template.html', prediction_text=prediction_text, form_visible=form_visible)

if __name__ == "__main__":
    app.run(debug=True)
