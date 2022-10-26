# importing Flask and other modules
import json
import os
import pandas as pd
import pickle
from google.cloud import storage
from flask import Flask, request, render_template

# Flask constructor
app = Flask(__name__)


# A decorator used to tell the application
# which URL is associated function
@app.route('/checkbodyfat', methods=["GET", "POST"])
def check_bodyfat():
    if request.method == "POST":
        prediction_input = [
            {
                "Age": int(request.form.get("Age")),
                "Neck": int(request.form.get("Neck")),
                "Knee": int(request.form.get("Knee")),
                "Ankle": int(request.form.get("Ankle")),
                "Biceps": int(request.form.get("Biceps")),
                "Forearm": float(request.form.get("Forearm")),
                "Wrist": float(request.form.get("Wrist")),
                "Weight": float(request.form.get("Weight")),
                "Height": float(request.form.get("Height")),
                "Abdomen": float(request.form.get("Abdomen")),
                "Chest": float(request.form.get("Chest")),
                "Hip": float(request.form.get("Hip")),
                "Thigh": int(request.form.get("Thigh"))
            }
        ]
        print(prediction_input)
        # Importing model from the pipeline bucket
        storage_client = storage.Client(project='de2022-362617')

        bucket = storage_client.bucket('bodyfat_model')
        blob = bucket.blob('model.pkl')
        filename = '/tmp/local_model.pkl'
        blob.download_to_filename(filename)
        blob_t = bucket.blob('transformer.pkl')
        filename_t = '/tmp/transformer.pkl'
        blob_t.download_to_filename(filename_t)

        model = pickle.load(open(filename, 'rb'))
        transformer = pickle.load(open(filename_t, 'rb'))

        X = pd.DataFrame.from_dict(prediction_input)

        X['Bmi'] = 703 * X['Weight'] / (X['Height'] * X['Height'])
        X['ACratio'] = X['Abdomen'] / X['Chest']
        X['HTratio'] = X['Hip'] / X['Thigh']
        X.drop(['Weight', 'Height', 'Abdomen', 'Chest', 'Hip', 'Thigh'], axis=1, inplace=True)

        # Transformer
        X = transformer.transform(X)
        density = model.predict(X)
        fat = ((4.95 / density[0]) - 4.5) * 100
        return {'Used model': model, 'Density': density[0], 'Bodyfat': fat}
    return render_template(
        "user_form.html")  # this method is called of HTTP method is GET, e.g., when browsing the link


if __name__ == '__main__':
    # app.run(host='0.0.0.0', port=5001)
    app.run(host='0.0.0.0', port=5000)
