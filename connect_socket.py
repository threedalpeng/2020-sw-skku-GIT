import socketio
import cv2


class SocketClient:
    sio = socketio.Client()

    def __init__(self, url):
        self.sio.connect(url)

    def emit(self, event, data):
        self.sio.emit(event, data)

    def wait(self):
        self.sio.wait()

    def send_data(self, img, data_dict, format=".png"):
        img_bin = self._ndarray2img(img, format=format)
        data_dict["img"] = img_bin
        socket.emit("msg", data_dict)

    @staticmethod
    @sio.event
    def connect():
        print("connection established")

    @staticmethod
    @sio.on("connect_success")
    def on_message(data):
        print(data)

    @staticmethod
    def _ndarray2img(img_array, format=".png"):
        return cv2.imencode(format, img_array)[1].tostring()


if __name__ == "__main__":

    # create Socket
    socket = SocketClient("http://localhost:3000")

    # get image
    img = cv2.imread("./img.png")

    # send data (image, data dictionary)
    socket.send_data(img, {"msg": "hi"})

    # wait socket event
    socket.wait()
