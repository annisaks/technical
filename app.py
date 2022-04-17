from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)


@app.route("/", methods=['GET', 'POST'])
def weight_prediction():
    if request.method == 'GET':
        return render_template("home.html")
    elif request.method == 'POST':
        gender = request.form.get("gender")
        height = request.form.get("height")
        input = [[gender, height]]
        input = np.array(input)
        # print(dict(request.form))
        # input = dict(request.form).values()
        # input = np.array([float(x) for x in input])
        # data = input.reshape(-1, 1)
        model = joblib.load(
            "model-development/model.pkl")
        # height = std_scaler.transform([height])
        result = model.predict(input)
        result = f'Your weight is {(result[0]): .2f} kg'
        return render_template('home.html', result=result)
    else:
        return "Unsupported Request Method"


if __name__ == '__main__':
    app.run(port=5000, debug=True)
