from flask import Flask, Response
import cv2
import numpy as np
import tensorflow as tf
import time
from gpiozero import OutputDevice

# =========================
# LOAD MODEL

# =========================
model = tf.keras.models.load_model(
    "new_waste_classifier_mobilenetv2.keras"
)

preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

IMG_SIZE = (224, 224)


# =========================
# CONFIG
# =========================
NUM_FRAMES = 10
FRAME_DELAY = 0.1
PREDICTION_INTERVAL = 3

last_prediction_time = 0

label = "Waiting..."
confidence = 0.0

# =========================
# GPIO SETUP
# =========================

ORGANIC_PIN = OutputDevice(17)
RECYCLE_PIN = OutputDevice(18)

# =========================
# CAMERA
# =========================
cap = cv2.VideoCapture(0)

# =========================
# FLASK APP
# =========================
app = Flask(__name__)

# =========================
# TEMPORAL AVERAGING
# =========================
def predict_average(frame):

    preds = []

    for _ in range(NUM_FRAMES):

        img = cv2.resize(frame, IMG_SIZE)

        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img.astype(np.float32))

        pred = model.predict(img, verbose=0)[0][0]

        preds.append(pred)

        time.sleep(FRAME_DELAY)

    return np.mean(preds)

# =========================
# VIDEO STREAM
# =========================
def generate_frames():

    global last_prediction_time
    global label
    global confidence

    while True:

        success, frame = cap.read()

        if not success:
            break

        frame = cv2.flip(frame, 1)

        h, w, _ = frame.shape

        # ROI BOX
        cv2.rectangle(
            frame,
            (w//4, h//4),
            (3*w//4, 3*h//4),
            (255, 0, 0),
            2
        )

        current_time = time.time()

        # =========================
        # PREDICTION
        # =========================
        if current_time - last_prediction_time > PREDICTION_INTERVAL:

            crop = frame[h//4:3*h//4, w//4:3*w//4]

            avg_pred = predict_average(crop)

            if avg_pred < 0.5:
                label = "Organic"
                confidence = 1 - avg_pred
                final_output = "O"
                 # Trigger Organic pin
                ORGANIC_PIN.on()
                RECYCLE_PIN.off()

                print("\n===== ORGANIC DETECTED =====")
                print(f"GPIO17: {ORGANIC_PIN.value}")
                print(f"GPIO18: {RECYCLE_PIN.value}")
                
                time.sleep(0.1)

                ORGANIC_PIN.off()

            else:
                label = "Recyclable"
                confidence = avg_pred
                final_output = "R"
                # Trigger Recycle pin
                ORGANIC_PIN.off()
                RECYCLE_PIN.on()

                print("\n===== RECYCLABLE DETECTED =====")
                print(f"GPIO17: {ORGANIC_PIN.value}")
                print(f"GPIO18: {RECYCLE_PIN.value}")

                time.sleep(0.1)

                RECYCLE_PIN.off()

            # print(f"{label}: {confidence:.2f}")

            # UART to PIC here

            last_prediction_time = current_time

        # =========================
        # DISPLAY TEXT
        # =========================
        text = f"{label}: {confidence:.2f}"

        cv2.putText(
            frame,
            text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)

        frame = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n'
            + frame +
            b'\r\n'
        )

# =========================
# ROUTE
# =========================
@app.route('/video')

def video():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

# =========================
# MAIN
# =========================
if __name__ == "__main__":

    app.run(
        host='0.0.0.0',
        port=5000,
        threaded=True
    )