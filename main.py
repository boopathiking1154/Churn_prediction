from flask import Flask, render_template, request, redirect, url_for, send_file, flash
import os
# we are importing the function that makes predictions.
from churn_model import predict_churn
import pandas as pd

app = Flask(__name__,template_folder='templates')
app.config["DEBUG"] = True
app.secret_key = "abc"

# we are setting the name of our html file that we added in the templates folder
@app.route("/")
def index():
    return render_template('index.html')


# this is how we are getting the file that the user uploads.
# then we are setting the path that we want to save it so we can use it later for predictions
@app.route("/", methods=['GET','POST'])
def uploadFiles():
    if request.method == 'POST':

        uploaded_file = request.files['file']

        if uploaded_file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if uploaded_file.filename != '':
            file_path = ("static/files/file.csv")
            uploaded_file.save(file_path)
    return redirect(url_for('downloadFile'))


# now we are reading the file, make predictions with our model and save the predictions.
# then we are sending the CSV with the predictions to the user as attachement
@app.route('/download')
def downloadFile():
    path = "static/files/file.csv"
    predictions = predict_churn(pd.read_csv(path))
    predictions.to_csv('static/files/predictions.csv', index=False)
    return send_file("static/files/predictions.csv", as_attachment=True)


# here we are setting the port.
if (__name__ == "__main__"):
    app.run(debug=True, host='0.0.0.0', port=9010)
