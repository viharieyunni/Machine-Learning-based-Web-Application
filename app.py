from flask import Flask, render_template, request, url_for, redirect
from send_mail import send_mail
import numpy as np
import pickle


app = Flask(__name__)

model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/submit', methods=['POST','GET'])
def submit():
    if request.method == 'POST' or request.method == 'GET':
        int_features = [int(x) for x in request.form.values()]
        final = [np.array(int_features)]
        #print(int_features)
        #print(final)
        prediction = model.predict_proba(final)
        output = '{0:.{1}f}'.format(prediction[0][1], 2)
        if output > str(0.5):
            return render_template('success.html',pred='Your heart is in Danger.\nProbability of heart attack occuring is {}'.format(output))
        else:
            return render_template('success.html',pred='Your heart is safe.\n Probability of heart attack occuring is {}'.format(output))
    return render_template('success.html')
if __name__ == '__main__':
    app.run(debug=True)
