import logging
import os
import typing
from flask import Flask, request, jsonify
import Main
import secrets
from werkzeug.utils import secure_filename

#Below is a Flask server that runs on port 8000 and uses the Battlesnake API to respond to requests from play.battlesnake.com:
def create_app(handlers: typing.Dict):
    app = Flask("Battlesnake")

    #For an "info" request:
    @app.get("/")
    def on_info():
        return handlers["info"]()

    #For a "start" request:
    @app.post("/start")
    def on_start():
        game_state = request.get_json()
        handlers["start"](game_state)
        return "ok"

    #For a "move" request:
    @app.post("/move")
    def on_move():
        game_state = request.get_json()

        response = handlers["move"](game_state)
        return response

    #For an "end" request:
    @app.post("/end")
    def on_end():
        game_state = request.get_json()
        handlers["end"](game_state)
        return "ok"

    #Adding a header to the response object:
    @app.after_request
    def identify_server(response):
        response.headers.set("server",
                             "battlesnake/github/starter-snake-python")
        return response


    # Temporary endpoint for uploading the PyTorch checkpoint:
    @app.post("/upload-checkpoint")
    def upload_checkpoint():

        # Authenticate the upload:
        token = os.environ.get("UPLOAD_TOKEN")

        if not token:
            return jsonify({"error": "Upload token not configured"}), 500

        if request.headers.get("Authorization") != f"Bearer {token}":
            return jsonify({"error": "Unauthorized"}), 401

        # Check that a file was provided:
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]

        if file.filename != "checkpoint.pth":
            return jsonify({"error": "Invalid filename"}), 400

        # Save directly to the Railway persistent volume:
        destination = "/data/checkpoint.pth"

        os.makedirs("/data", exist_ok=True)

        file.save(destination)

        # Verify that the uploaded file is not empty:
        if os.path.getsize(destination) == 0:
            os.remove(destination)
            return jsonify({"error": "Uploaded file is empty"}), 400

        return jsonify({
            "message": "Checkpoint uploaded successfully",
            "size": os.path.getsize(destination)
        }), 200
    

    return app

Battlesnake = create_app({
    "info": Main.info,
    "start": Main.start,
    "move": Main.move,
    "end": Main.end
})
