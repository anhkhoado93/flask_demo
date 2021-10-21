from flask import Flask, jsonify, request
from translate import correct

app = Flask(__name__)


@app.route('/', methods=['POST'])
def index():
    if request.method == 'POST':
        print("ok",request.get_json())
        data = request.get_json()['incorrect']
        response = ""
        if data != "":
            ## Do Something
            print(data)
            response = correct(str(data))

        return jsonify({'correct': response})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)


