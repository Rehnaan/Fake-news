from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__)

model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    news = request.form["news"]

    transformed_news = vectorizer.transform([news])

    prediction = model.predict(transformed_news)

    if prediction[0] == 0:
        result = "Fake News"
    else:
        result = "Real News"

    return render_template("index.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)