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
        
    return app

Battlesnake = create_app({
    "info": Main.info,
    "start": Main.start,
    "move": Main.move,
    "end": Main.end
})
