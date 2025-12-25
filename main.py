import sys      # For exiting the program and printing errors
import os       # To check for model file existence (os.path.exists)
import time     # Used for sound cooldown and delays
import platform # To detect operating system (Windows, macOS, Linux)
import subprocess  # To run system-level sound commands (on macOS)
import cv2      # OpenCV: For video capture and drawing UI elements
import numpy as np  # For numerical operations
from tensorflow.keras.models import load_model  # To load the trained fire detection model (.h5 file)
from tensorflow.keras.preprocessing import image  # To preprocess frames


def _play_mac_beep(alert_type="alert"):
    sounds = {
        'beep': ["Pop"],
        'alert': ["Ping"],
        'warning': ["Basso"],
        'alarm': ["Funk"]
    }.get(alert_type, ["Ping"])
    for sound in sounds:
        subprocess.Popen(["afplay", f"/System/Library/Sounds/{sound}.aiff"])
        time.sleep(0.2)


try:
    import winsound

    SOUND_SYSTEM = 'windows'
except ImportError:
    try:
        import pygame

        pygame.mixer.init()
        SOUND_SYSTEM = 'pygame'
    except ImportError:
        SOUND_SYSTEM = 'none'
        print("⚠️ Audio system not available.")


def open_camera(index=0):
    system = platform.system()
    backends = {
        'Darwin': [cv2.CAP_AVFOUNDATION, cv2.CAP_QT],
        'Windows': [cv2.CAP_DSHOW, cv2.CAP_MSMF],
        'Linux': [cv2.CAP_V4L2]
    }.get(system, [cv2.CAP_ANY])
    for backend in backends:
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            return cap
    raise IOError("Error: Could not access camera.")



class FireDetector:
    def __init__(self, model_path="fire_model.h5"):
        self.model = self._load_model(model_path)
        self.img_size = (224, 224)
        self.last_alert = 0
        self.levels = {
            'no_fire': {'range': (0.0, 0.35), 'color': (0, 255, 0), 'sound': None},
            'low': {'range': (0.35, 0.5), 'color': (0, 255, 255), 'sound': None},
            'medium': {'range': (0.5, 0.7), 'color': (0, 165, 255), 'sound': 'alert'},
            'high': {'range': (0.7, 0.9), 'color': (0, 100, 255), 'sound': 'warning'},
            'critical': {'range': (0.9, 1.0), 'color': (0, 0, 255), 'sound': 'alarm'}
        }

    def _load_model(self, path):
        if not os.path.exists(path):
            print(f"Model file not found: {path}")
            sys.exit(1)
        try:
            model = load_model(path)
            print("✅ Model loaded successfully.")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)

    def preprocess_frame(self, frame):
        resized = cv2.resize(frame, self.img_size)
        arr = image.img_to_array(resized)
        return np.expand_dims(arr, axis=0) / 255.0

    def predict_fire(self, frame):
        # Main prediction
        main_conf = self.model.predict(self.preprocess_frame(frame))[0][0]

        # Regional prediction
        h, w = frame.shape[:2]
        regions = [
            frame[:h // 2, :w // 2],
            frame[:h // 2, w // 2:],
            frame[h // 2:, :w // 2],
            frame[h // 2:, w // 2:],
            frame[h // 4:3 * h // 4, w // 4:3 * w // 4]
        ]
        region_conf = []
        for region in regions:
            try:
                conf = self.model.predict(self.preprocess_frame(region))[0][0]
                region_conf.append(conf)
            except:
                pass
        return max([main_conf] + region_conf)

    def get_fire_level(self, confidence):
        for name, data in self.levels.items():
            if data['range'][0] <= confidence <= data['range'][1]:
                data['name'] = name
                return name, data
        return 'critical', self.levels['critical']

    def get_fire_level_name(self, confidence):
        name, _ = self.get_fire_level(confidence)
        return name

    def draw_detection_box(self, frame, confidence, level_data):
        h, w = frame.shape[:2]
        if confidence > 0.3:
            margin = int(w * 0.15)
            cv2.rectangle(frame, (margin, margin), (w - margin, h - margin),
                          level_data['color'], max(2, int(confidence * 5)))

    def draw_info_panel(self, frame, confidence, level_data):  # [SULTANBI]
        h, w = frame.shape[:2]
        color = (255, 255, 255) if confidence < 0.7 else level_data['color']
        cv2.rectangle(frame, (0, 0), (w, 100), (0, 0, 0), -1)
        cv2.putText(frame, f"Level: {level_data['name'].upper()}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.putText(frame, f"Confidence: {confidence:.1%}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    def play_sound_alert(self, level_name):
        if time.time() - self.last_alert < 2:
            return
        sound = self.levels[level_name]['sound']
        if not sound:
            return
        try:
            if SOUND_SYSTEM == 'windows':
                winsound.Beep(1000, 300)
            elif SOUND_SYSTEM == 'pygame':
                pass
            elif platform.system() == 'Darwin':
                _play_mac_beep(sound)
        except:
            print("⚠️ Sound error")
        self.last_alert = time.time()

    def run_detection(self):
        print("🔥 PYRO-Vision Fire Detection Started")
        print("Enter 'webcam' or video file path:")
        source = input("Source: ").strip()

        if source.lower() in ['webcam', '0']:
            cap = open_camera(0)
        else:
            cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            print("❌ Could not open video source.")
            return

        cv2.namedWindow("🔥 Fire Detection", cv2.WINDOW_NORMAL)

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("📷 Frame error or stream ended.")
                    break

                confidence = self.predict_fire(frame)
                level_name, level_data = self.get_fire_level(confidence)

                self.draw_info_panel(frame, confidence, level_data)
                self.draw_detection_box(frame, confidence, level_data)
                self.play_sound_alert(level_name)

                cv2.imshow("🔥 Fire Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("🛑 Exit requested.")
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("✅ Detection stopped.")


def main():
    detector = FireDetector()
    detector.run_detection()


if __name__ == "__main__":
    main()
