import cv2 as cv
import numpy as np

class CameraIO:
    def __init__(self, camera_index=1, width=640, height=480):
        """
        High-performance Camera I/O for Signal Processing applications.
        Strictly utilizes DirectShow for hardware-level compatibility.
        """
        self.cap = cv.VideoCapture(camera_index, cv.CAP_DSHOW)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Hardware failure: Camera at index {camera_index} unreachable.")

        # Latency optimization
        self.cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)
        
        self.actual_w = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))
        self.actual_h = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        
        print(f"[*] Stream synchronized: {self.actual_w}x{self.actual_h}")

    def get_frame(self):
        """Retrieves raw frame data from the stream."""
        ret, frame = self.cap.read()
        return frame if ret else None

    def release(self):
        """Implicit resource deallocation."""
        if self.cap.isOpened():
            self.cap.release()

def get_available_indices(limit=5):
    """Scan and map available hardware video nodes."""
    nodes = []
    for i in range(limit):
        temp = cv.VideoCapture(i, cv.CAP_DSHOW)
        if temp.isOpened():
            nodes.append(i)
            temp.release()
    return nodes

if __name__ == "__main__":
    indices = get_available_indices()
    if not indices:
        print("[!] Error: No imaging hardware detected.")
        exit(1)

    # Instantiate with the verified hardware index
    cam = CameraIO(camera_index=indices[-1])
    
    try:
        while True:
            frame = cam.get_frame()
            if frame is None: break
            cv.imshow("Signal Acquisition Buffer", frame)
            if cv.waitKey(1) & 0xFF == ord('q'): break
    finally:
        cam.release()
        cv.destroyAllWindows()