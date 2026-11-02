from flask import Flask, render_template, request
from transformers import pipeline

app = Flask(__name__)

classifier = pipeline(
    "text-classification", model = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)


@app.route("/", methods=["GET", "POST"])
def index():
    text = ""
    result = None
    if request.method == "POST":
        text = request.form.get("review","").strip()
    if text:
        prediction =  classifier(text, truncation=True, max_length=512)[0]
        result = {"label": prediction["label"], "score":prediction["score"]}

    return render_template("index.html", text=text, result=result)

if __name__ == "__main__":
    app.run(debug=True)
