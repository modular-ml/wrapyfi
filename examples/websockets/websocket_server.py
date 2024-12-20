from flask import Flask
from flask_socketio import SocketIO, Namespace
import logging
import base64

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


@socketio.event(namespace=WEBSOCKET_NAMESPACE)
def connect():
    """
    This is the default event handler for when a client connects to the server.
    """
    print("Connected to the server")


@socketio.on("/hello/my_message", namespace=WEBSOCKET_NAMESPACE)
def my_text_message(data):
    """
    This is for the `websockets/publisher_client.py` example for when the publisher sends a message over Websockets.
    To receive the message, the `websockets/listener_client.py` script must be running.

    :param data: dict: The message data
    """
    print("Received 'my_message' in client script")
    socketio.emit("/hello/my_message", data, namespace=WEBSOCKET_NAMESPACE)


@socketio.on("/cam_mic/cam_feed", namespace=WEBSOCKET_NAMESPACE)
def my_img_message(data):
    """
    This is for the `sensors/cam_mic.py` example for when the publisher sends an image message over websocket (setting `--mware` argument to 'websocket').

    :param data: dict: The image data
    """
    print("Received 'my_message' in client script")
    socketio.emit("/cam_mic/cam_feed", data, namespace=WEBSOCKET_NAMESPACE)


@socketio.on("/cam_mic/audio_feed", namespace=WEBSOCKET_NAMESPACE)
def my_aud_message(data):
    """
    This is for the `sensors/cam_mic.py` example when the publisher sends an audio message over websocket (setting `--mware` argument to 'websocket').

    :param data: dict: The audio data
    """
    print("Received 'my_message' in client script")
    socketio.emit("/cam_mic/audio_feed", data, namespace=WEBSOCKET_NAMESPACE)


# TODO (fabawi): additional forwarders for debugging purposes. To be removed soon


@socketio.on("/camera/effect_image", namespace=WEBSOCKET_NAMESPACE)
def my_imgefct_message(data):
    """
    Handle effect image data, ensure Base64 encoding for transport.
    """
    try:
        socketio.emit("/camera/effect_image", data, namespace=WEBSOCKET_NAMESPACE)
    except Exception as e:
        logging.error(f"Error in /camera/effect_image: {e}")


@socketio.on("/camera/raw_image", namespace=WEBSOCKET_NAMESPACE)
def my_imgraw_message(data):
    """
    Handle raw image data, ensure Base64 encoding for transport.
    """
    try:
        socketio.emit("/camera/raw_image", data, namespace=WEBSOCKET_NAMESPACE)
    except Exception as e:
        logging.error(f"Error in /camera/raw_image: {e}")


@socketio.on("/message/my_message_snd", namespace=WEBSOCKET_NAMESPACE)
def my_mtrcssnd_message(data):
    """
    Handle 'my_message_snd', ensure Base64 encoding if data is binary.
    """
    try:
        socketio.emit("/message/my_message_snd", data, namespace=WEBSOCKET_NAMESPACE)
    except Exception as e:
        logging.error(f"Error in /message/my_message_snd: {e}")


@socketio.on("/message/my_message_rec", namespace=WEBSOCKET_NAMESPACE)
def my_mtrcsrec_message(data):
    """
    Handle 'my_message_rec', ensure Base64 encoding if data is binary.
    """
    try:
        socketio.emit("/message/my_message_rec", data, namespace=WEBSOCKET_NAMESPACE)
    except Exception as e:
        logging.error(f"Error in /message/my_message_rec: {e}")


# Start the development server. Note that for a production deployment, a production-ready server like waitress should be used.
if __name__ == "__main__":
    socketio.run(
        app, host=SOCKET_IP, port=SOCKET_PORT, debug=True, allow_unsafe_werkzeug=True
    )
