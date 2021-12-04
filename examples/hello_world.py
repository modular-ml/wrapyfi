from wrapify.connect.wrapper import MiddlewareCommunicator


class HelloWorld(MiddlewareCommunicator):
    @MiddlewareCommunicator.register("NativeObject", "yarp", "HelloWorld", "/hello/my_message",
                                     carrier="", should_wait=True)
    def send_message(self):
        msg = input("Type your message: ")
        obj = {"message": msg}
        return obj,


hello_world = HelloWorld()

LISTEN = False
hello_world.activate_communication("send_message", mode="listen" if LISTEN else "publish")

while True:
    my_message, = hello_world.send_message()
    print(my_message["message"])
