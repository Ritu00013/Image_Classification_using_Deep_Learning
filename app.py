from flask import Flask, render_template

app = Flask(__name__)

@app.route("/", methods=["POST", "GET"])
def home():
    return render_template("index.html")

@app.route("/chooseModel", methods=["POST", "GET"])
def chooseModel():
    return render_template("choose.html")

if __name__ == "__main__":
    app.run(debug=True)