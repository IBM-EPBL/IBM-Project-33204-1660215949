from flask import Flask, render_template, request
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import session
from flask import Flask, redirect, url_for
import requests
import json
import ibm_db

app = Flask(__name__, template_folder="templates")
app.secret_key = 'NutritionAnalyzer'
model = load_model('NutritionAnalyser.h5')

def connectToDB():
    try:
        connection = ibm_db.connect("DATABASE=bludb;\
        HOSTNAME=125f9f61-9715-46f9-9399-c8177b21803b.c1ogj3sd0tgtu0lqde00.databases.appdomain.cloud;\
        PORT=30426;\
        Security=SSL;\
        SSLServerCertificate=DigiCertGlobalRootCA.crt;\
        UID=qlj81410;\
        PWD=phBPVWNuoifGiYIC;", "", "")
        print("Connected to DB!")
        return connection
    except:
        print("error while connecting ", ibm_db.conn_errormsg())
        return 0


connection = connectToDB()

@app.route('/Classify')
def index():
    return render_template('classify.html')

@app.route("/")
@app.route('/home')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname('__file__')
        filepath = os.path.join(basepath, "test", f.filename)
        f.save(filepath)

        img = image.load_img(filepath, target_size=(64, 64))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)

        pred = np.argmax(model.predict(x), axis=1)
        index = ['APPLES', 'BANANA', 'ORANGE', 'PINEAPPLE', 'WATERMELON']

        predictedValue = str(index[pred[0]])

        result = nutrition(predictedValue)


        temp = result.json()["items"]
        items = temp[0]
        print(items)
        sugar = items["sugar_g"]
        fiber = items["fiber_g"]
        sodium = items["sodium_mg"]
        potassium = items["potassium_mg"]
        fat_saturated = items["fat_saturated_g"]
        fat_total = items["fat_total_g"]
        calories = items["calories"]
        cholesterol = items["cholesterol_mg"]
        protein = items["protein_g"]
        carbohydrates = items["carbohydrates_total_g"]
        return render_template("result.html", name=(predictedValue), sugar=(sugar), fiber=(fiber), sodium=(sodium), potassium=(potassium), fat_saturated=(fat_saturated), fat_total=(fat_total), calories=(calories), cholesterol=(cholesterol), protein=(protein), carbohydrates=(carbohydrates))


def nutrition(index):

    url = "https://calorieninjas.p.rapidapi.com/v1/nutrition"

    querystring = {"query": index}

    headers = {
        'x-rapidapi-key': "5d797ab107mshe668f26bd044e64p1ffd34jsnf47bfa9a8ee4",
        'x-rapidapi-host': "calorieninjas.p.rapidapi.com"
    }

    response = requests.request(
        "GET", url, headers=headers, params=querystring)

    print("from api "+response.text)
    return response


if __name__ == "__main__":
    app.run(debug=False)
