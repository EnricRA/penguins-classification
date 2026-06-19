from flask import Flask, jsonify, request

from services.predict_utils import load_artifact, predict_species

app = Flask(__name__)
# Carga el artefacto del modelo SVM al iniciar el servicio
ARTIFACT = load_artifact("svm")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "svm"})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json(force=True)
        result = predict_species(ARTIFACT, payload)
        return jsonify({"model": "svm", **result})
    except ValueError as error:
        return jsonify({"error": str(error)}), 400
    except Exception as error:
        return jsonify({"error": str(error)}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5002, debug=False)
