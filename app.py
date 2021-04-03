from flask import Flask, render_template, request, redirect
import test
app = Flask(__name__)
@app.route("/")
def hello():
    return render_template("index.html")

@app.route("/", methods=['POST'])
def caption():
    if(request.method == "POST"):
        f = request.files["image"]
        path = "./static/{}".format(f.filename)
        f.save(path)
        caption = test.caption(path)
        # print(caption)
        result = {
            'image': path,
            'caption': caption
        }
    return render_template("index.html", result = result)

@app.route("/home")
def home():
    return redirect("/")

if __name__ == "__main__":
    # app.debug = True
    app.run(host="0.0.0.0")
