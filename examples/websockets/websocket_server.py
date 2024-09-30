from flask import Flask
from flask_socketio import SocketIO, Namespace
import logging

# Default configuration (should the values not be changed in the environment)
SOCKET_IP = "0.0.0.0"
SOCKET_PORT = 5000
WEBSOCKET_NAMESPACE = "/"

# Initialize Flask and Flask-SocketIO
app = Flask(__name__)
app.config["SECRET_KEY"] = "some secret key"
socketio = SocketIO(app, logger=True, engineio_logger=True)

# Setup logging
logging.basicConfig(level=logging.DEBUG)


# Create a custom namespace class
class MyCustomNamespace(Namespace):
    """
    This is a custom namespace class that extends the Flask-SocketIO Namespace class.
    This class is used to handle custom events and manage connected clients.
    """

    connected_sids = set()

    def on_connect(self, sid, environ, auth):
        logging.debug(f"Client connected: {sid}")
        self.connected_sids.add(sid)
        logging.debug(f"All connected sids: {self.connected_sids}")

    def on_disconnect(self, sid):
        logging.debug(f"Client disconnected: {sid}")
        self.connected_sids.discard(sid)
        logging.debug(f"All connected sids: {self.connected_sids}")

    def trigger_event(self, event, *args):
        logging.debug(f"Received event '{event}' with args: {args}")
        handler = getattr(self, "on_" + event, None)
        if handler is not None:
            return handler(*args)
        else:
            logging.warning(f"No handler for event '{event}'")


# Register the custom namespace
socketio.on_namespace(MyCustomNamespace(WEBSOCKET_NAMESPACE))


@socketio.event(namespace="/")
def connect():
    """
    This is the default event handler for when a client connects to the server.
    """
    print("Connected to the server")


@socketio.on("/hello/my_message", namespace="/")
def my_text_message(data):
    """
    This is for the `websockets/publisher_client.py` example for when the publisher sends a message over Websockets.
    To receive the message, the `websockets/listener_client.py` script must be running.

    :param data: dict: The message data
    """
    print("Received 'my_message' in client script")
    socketio.emit("/hello/my_message", data, namespace="/")


@socketio.on("/cam_mic/cam_feed", namespace="/")
def my_img_message(data):
    """
    This is for the `sensors/cam_mic.py` example for when the publisher sends an image message over websocket (setting `--mware` argument to 'websocket').

    :param data: dict: The image data
    """
    print("Received 'my_message' in client script")
    socketio.emit("/cam_mic/cam_feed", data, namespace="/")


@socketio.on("/cam_mic/audio_feed", namespace="/")
def my_aud_message(data):
    """
    This is for the `sensors/cam_mic.py` example when the publisher sends an audio message over websocket (setting `--mware` argument to 'websocket').

    :param data: dict: The audio data
    """
    print("Received 'my_message' in client script")
    socketio.emit("/cam_mic/audio_feed", data, namespace="/")


# Start the development server. Note that for a production deployment, a production-ready server like waitress should be used.
if __name__ == "__main__":
    socketio.run(
        app, host=SOCKET_IP, port=SOCKET_PORT, debug=True, allow_unsafe_werkzeug=True
    )
