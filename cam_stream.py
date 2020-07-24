import cv2
from threading import Thread, Lock
import time


class CamStream(Thread):
    def __init__(self, path, width=None, height=None, frame_rate=".30"):
        super(CamStream, self).__init__()
        self.setDaemon(True)
        self.cap = cv2.VideoCapture(path)
        if width is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)  # CAP_PROP_FRAME_WIDTH
        if height is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)   # CAP_PROP_FRAME_HEIGHT
        self.frame_rate = self.parse_frame(frame_rate)
        self.isRunning = False
        self.latestFrame = None
        self.lock = Lock()

    def start(self):
        print("start")
        with self.lock:
            while self.latestFrame is None:
                frame_ready, self.latestFrame = self.cap.read()
        self.isRunning = True
        print("ready")
        super().start()

    def run(self):
        while self.isRunning:
            with self.lock:
                frame_ready, latest_frame = self.cap.read()
                if latest_frame is not None:
                    # print('get')
                    self.latestFrame = latest_frame
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
        if self.latestFrame is not None:
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
    cap.set_frame_rate("00:01:00")  # 1분에 한번씩 출력
    # cap.set_frame_rate(".30")     # 30fps로 출력

    cap.start()

    # while not cap.is_ready():
    #    time.sleep(5)

    prev_time = time.time()
    elapsed = cap.get_frame_rate()
    after_show = False
    image_to_show = None
    while True:
        if after_show and (cv2.waitKey(20) & 0xFF == ord('q')):
            break

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
        after_show = True

    cap.stop()
