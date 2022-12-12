from collections import Counter
from functools import reduce
import json

import zmq

# Create a ZeroMQ context
context = zmq.Context()

# Connect to the parameter server

# Connect to the request server
request_server = context.socket(zmq.REQ)
request_server.connect("tcp://localhost:5556")

param_server = context.socket(zmq.SUB)
param_server.connect("tcp://127.0.0.1:5555")
# param_server.setsockopt_string(zmq.SUBSCRIBE, "foo")
param_server.setsockopt(zmq.RCVTIMEO, 2000)

RECHECK_COUNT = 5

default_command = "set foo/bar/42"


def access_nested_dict(d, keys):
    return reduce(lambda x,y: x[y], keys[:-1], d)[keys[-1]]


def parse_prefix(param_full: str, topics: dict):
    # Split the string by '/' to get the individual components
    components = param_full.split('/')
    # If there is only one component in the list, add it to the topics dictionary
    # as a value and return
    if len(components) == 1:
        topics[""] = components[0]
        return topics
    # Get the first component and remove it from the list
    key = components.pop(0)
    # If the key doesn't exist in the topics dictionary, add it as a key
    # and recursively call parse_prefix with the remaining components
    if key not in topics:
        topics[key] = parse_prefix("/".join(components), {})
    # If the key already exists in the topics dictionary, recursively call
    # parse_prefix with the remaining components and update the value of the
    # key in the topics dictionary
    else:
        topics[key] = parse_prefix("/".join(components), topics[key])
    # Return the updated topics dictionary
    return topics


def reverse_parse_prefix(topics: dict, prefix: str = ""):
    # Create an empty list to store the generated strings
    strings = []
    # Iterate over the keys in the topics dictionary
    for key in topics:
        # If the value of the key is a string, add the key and value to the list
        # of strings as a '/' separated string
        if isinstance(topics[key], str):
            strings.append(key + '/' + topics[key])
        # If the value of the key is a dictionary, recursively call generate_strings
        # with the value and add the returned list of strings to the list of strings
        # with the key as a prefix
        else:
            nested_strings = reverse_parse_prefix(topics[key])
            for string in nested_strings:
                strings.append(key + '/' + string)
    # Return the list of strings filtered for repeated /'s
    return [prefix + string.replace('//', '/') for string in strings]


while True:
    # Send a request to the request server
    new_command = str(input(f"input command: (default - {default_command})")) or default_command
    request_server.send_string(new_command)
    default_command = new_command

    response = request_server.recv_string()
    print("Received response from server: %s" % response)
    if "success:::" in response:
        topics = {}
        current_prefix = response.split(":::")[1].encode('utf-8')
        param_server.subscribe(current_prefix)
        # time.sleep(0.2)
        prev_params = Counter()
        param = "!"
        while True:
            # Receive updates from the parameter server
            prefix, param, value = param_server.recv_multipart()
            # Construct the full parameter name with the namespace prefix
            prefix, param, value = prefix.decode('utf-8'), param.decode('utf-8'), value.decode('utf-8')
            if (param in prev_params or param is None) and prev_params[param] == RECHECK_COUNT:
                break
            prev_params[param] += 1
            full_param = "/".join([prefix, param, value])
            parse_prefix(full_param, topics)
            # print("Received update: %s" % (full_param))
        topic_results = access_nested_dict(topics, current_prefix.decode('utf-8').split('/'))
        print(json.dumps(topic_results, indent=2, default=str))
        print("Reverse parse:")
        print(reverse_parse_prefix(topic_results, prefix=f"set {current_prefix.decode('utf-8')}/"))
        # Close the connection
        param_server.unsubscribe(current_prefix)



