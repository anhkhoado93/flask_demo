from flask import Flask, jsonify, request
from translate import correct
app = Flask(__name__)

@app.route('/', methods=['POST'])
def index():
    if request.method == 'POST':
        data = request.data
        response = ""
        if data != "":
            ## Do Something
            response = correct(data)
        
        return jsonify({'correct': response})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)