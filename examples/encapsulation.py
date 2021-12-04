from wrapify.connect.wrapper import MiddlewareCommunicator


class Encapsulator(MiddlewareCommunicator):
    @MiddlewareCommunicator.register("NativeObject", "yarp", "Encapsulator", "/encapsulator/my_message_modifier",
                                     carrier="", should_wait=True)
    def encapsulated_modify_message(self, msg):
        msg = "******" + msg + "******"
        return msg,

    @MiddlewareCommunicator.register("NativeObject", "yarp", "Encapsulator", "/encapsulator/my_message",
                                     carrier="", should_wait=True)
    def encapsulating_send_message(self):
        msg = input("Type your message: ")
        msg, = self.encapsulated_modify_message(msg)
        obj = {"message": msg}
        return obj,


encapsulator = Encapsulator()

LISTEN = False
encapsulator.activate_communication("encapsulating_send_message", mode="listen" if LISTEN else "publish")
encapsulator.activate_communication("encapsulated_modify_message", mode="listen" if LISTEN else "publish")

while True:
    my_message, = encapsulator.encapsulating_send_message()
    print(my_message["message"])
