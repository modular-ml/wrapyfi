import zmq
import json
import argparse
import logging

logging.getLogger().setLevel(logging.INFO)


def monitor_active_connections(socket_pub_address, topic):
    try:
        context = zmq.Context()
        subscriber = context.socket(zmq.SUB)

        subscriber.connect(socket_pub_address)
        subscriber.setsockopt_string(zmq.SUBSCRIBE, topic)

        while True:
            topic, message = subscriber.recv_multipart()
            topic = topic.decode("utf-8")
            data = json.loads(message.decode("utf-8"))
            logging.info(f"[ZeroMQ] Topic: {topic}, Data: {data}")

    except Exception as e:
        logging.error(f"[ZeroMQ] An error occurred: {str(e)}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--socket_pub_address",
        type=str,
        default="tcp://127.0.0.1:5555",
        help="Socket subscription address",
    )
    parser.add_argument(
        "--topic",
        type=str,
        default="ZEROMQ/CONNECTIONS",
        help="Topic to subscribe. The ZEROMQ/CONNECTIONS topic monitors subscribers to a any topic on "
        "the sub_address. Note that zeromq_proxy_broker.py in pubsub mode (either as a standalone"
        "or spawned by default by any publisher) must be running on socket_pub_address",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    monitor_active_connections(args.socket_pub_address, args.topic)
