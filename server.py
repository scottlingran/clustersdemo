from flask import Flask, jsonify, request
import wordy
import traceback
import os
from flask.ext.cors import CORS

app = Flask(__name__)
cors = CORS(app)

@app.route("/")
def hello():
    try:
        is_similar_query = request.args.getlist("list") is None or len(request.args.getlist("list")) == 0
        if is_similar_query:
            answer = wordy.get_similar(request.args)
        else:
            answer = wordy.get_list(request.args)
        res = jsonify({'data': answer})

        return (res, 200)
    except Exception, e:
        traceback.print_exc()
        return jsonify({'message': str(e)}), 500


if __name__ == "__main__":
    host = '0.0.0.0'
    port = 8000
    app.run(host=host, port=port, threaded=True)
