from flask import Flask, jsonify, request

from services.predict_utils import load_artifact, predict_species

app = Flask(__name__)
# Carga el artefacto del modelo k-NN al iniciar el servicio
ARTIFACT = load_artifact("knn")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "knn"})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json(force=True)
        result = predict_species(ARTIFACT, payload)
        return jsonify({"model": "knn", **result})
    except ValueError as error:
        return jsonify({"error": str(error)}), 400
    except Exception as error:
        return jsonify({"error": str(error)}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5004, debug=False)
