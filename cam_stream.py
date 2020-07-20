import cv2
from threading import Thread, Lock
import time


class CamStream(Thread):
    def __init__(self, path=0, width=1280, height=720, frame_rate=".30"):
        super(CamStream, self).__init__()
        self.setDaemon(True)
        self.cap = cv2.VideoCapture(path, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)  # CAP_PROP_FRAME_WIDTH
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)   # CAP_PROP_FRAME_HEIGHT\
        self.frame_rate = self.parse_frame(frame_rate)
        self.isRunning = False
        self.latestFrame = None
        self.frameReady = False
        self.lock = Lock()

    def start(self):
        with self.lock:
            while not self.frameReady:
                self.frameReady, self.latestFrame = self.cap.read()
        self.isRunning = True
        super().start()

    def run(self):
        while self.isRunning:
            with self.lock:
                self.frameReady, latestFrame = self.cap.read()
                if self.frameReady:
                    self.latestFrame = latestFrame
        super().run()

    def set_frame_rate(self, duration):
        self.frame_rate = self.parse_frame(duration)

    def get_frame_rate(self):
        return self.frame_rate

    @staticmethod
    def parse_frame(duration):
        if duration[0] == '.':
            return 1. / float(duration[1:])
        hms = duration.split(":")
        if len(hms) > 3 or len(hms) < 1:
            print("Error: Format must be 'hh:mm:ss', 'mm:ss', 'ss' or '.fps'")
            return 30
        elif len(hms) == 2:
            hms = ['00'] + hms
        elif len(hms) == 1:
            hms = ['00', '00'] + hms
        print(hms)
        return float(hms[0])*3600 + float(hms[1])*60 + float(hms[2])

    def get_latest_frame(self):
        if self.frameReady:
            return self.latestFrame
        elif self.latestFrame is not None:
            return self.latestFrame
        else:
            #print("Error!")
            return None

    def stop(self):
        self.isRunning = False
        self.cap.release()


# Example
if __name__ == "__main__":
    cap = CamStream(path=0)
    cap.set_frame_rate("00:00:08") # 8초에 한번씩 출력

    cap.start()

    # while not cap.is_ready():
    #    time.sleep(5)

    prev_time = time.time()
    elapsed = cap.get_frame_rate()
    image_to_show = None
    while True:
        im = cap.get_latest_frame()
        if elapsed >= cap.get_frame_rate():
            print(elapsed)
            image_to_show = im
        else:
            elapsed = time.time() - prev_time
            continue

        if image_to_show is None:
            continue

        prev_time = time.time()

        ###############################
        # do something costly in here #
        ###############################

        cv2.imshow("demo", image_to_show)
        elapsed = time.time() - prev_time
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    cap.stop()
