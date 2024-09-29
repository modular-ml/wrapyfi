from flask import Flask
from flask_socketio import SocketIO, Namespace
import logging

# Configuration
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
    print("Connected to the server")


@socketio.on("/hello/my_message", namespace="/")
def my_message(data):
    print("Received 'my_message' in client script")
    socketio.emit("/hello/my_message", data, namespace="/")


# Start the server
if __name__ == "__main__":
    socketio.run(
        app, host=SOCKET_IP, port=SOCKET_PORT, debug=True, allow_unsafe_werkzeug=True
    )
