from ultralytics import YOLO
import cv2, os, time, csv
from datetime import datetime
import easyocr

# ---------------- CONFIG ----------------
VIDEO_PATH = r"videos/police.mp4"
VEHICLE_MODEL_PATH = r"models/Vehicle-detect.pt"
PLATE_MODEL_PATH = r"models/plate.pt"  # license plate detection model

WIDTH, HEIGHT = 1280, 720
CROP_Y_START = 240
MARK1, MARK2 = 150, 400
MARK_GAP_METERS = 15
SPEED_LIMIT_KMPH = 20

os.makedirs('overspeeding/cars', exist_ok=True)
os.makedirs('overspeeding/plates', exist_ok=True)

# ---------------- LOAD MODELS ----------------
vehicle_model = YOLO(VEHICLE_MODEL_PATH)
plate_model = YOLO(PLATE_MODEL_PATH)
reader = easyocr.Reader(['en'])

VEHICLE_CLASSES = ['car', 'truck', 'bus', 'motorcycle', 'bicycle']

# ---------------- HELPERS ----------------
def save_vehicle_image(image):
    path = f'overspeeding/cars/{datetime.now().strftime("%Y%m%d_%H%M%S_%f")}.jpg'
    cv2.imwrite(path, image)
    return path

def save_plate_image(image):
    path = f'overspeeding/plates/{datetime.now().strftime("%Y%m%d_%H%M%S_%f")}.jpg'
    cv2.imwrite(path, image)
    return path

def estimate_speed(start_time, end_time):
    duration = end_time - start_time
    if duration <= 0:
        return 0
    return round((MARK_GAP_METERS / duration) * 3.6, 2)

def log_to_csv(vehicle_id, speed, plate_number, vehicle_img, plate_img):
    file_exists = os.path.isfile("overspeeding_log.csv")
    with open("overspeeding_log.csv", "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["VehicleID", "Speed(km/h)", "PlateNumber", "VehicleImage", "PlateImage"])
        writer.writerow([vehicle_id, speed, plate_number, vehicle_img, plate_img])

# ---------------- MAIN PROCESS ----------------
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Could not open video")
        return

    cv2.namedWindow("Vehicle Speed & Plate Detection", cv2.WINDOW_NORMAL)

    start_times, end_times, vehicle_speeds = {}, {}, {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize & crop
        frame = cv2.resize(frame, (WIDTH, HEIGHT))[CROP_Y_START:HEIGHT, 0:WIDTH]
        display = frame.copy()

        # Vehicle tracking
        results = vehicle_model.track(frame, persist=True, verbose=False)[0]

        if results.boxes.id is None:
            cv2.imshow("Vehicle Speed & Plate Detection", display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        for box, track_id in zip(results.boxes, results.boxes.id.int().cpu().tolist()):
            cls = int(box.cls[0])
            label = vehicle_model.names[cls]
            if label not in VEHICLE_CLASSES:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            bottom_y = y2

            # Draw vehicle bounding box
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Speed calculation
            if track_id not in start_times and MARK2 > bottom_y > MARK1 and y1 < MARK1:
                start_times[track_id] = time.time()

            elif track_id in start_times and track_id not in end_times and bottom_y > MARK2:
                end_times[track_id] = time.time()
                speed = estimate_speed(start_times[track_id], end_times[track_id])
                vehicle_speeds[track_id] = speed

                if speed > SPEED_LIMIT_KMPH:
                    cropped_vehicle = frame[y1:y2, x1:x2]
                    vehicle_img_path = save_vehicle_image(cropped_vehicle)

                    # License plate detection
                    plate_results = plate_model.predict(cropped_vehicle, verbose=False)[0]
                    plate_number, plate_img_path = "UNKNOWN", ""

                    for pbox in plate_results.boxes:
                        px1, py1, px2, py2 = map(int, pbox.xyxy[0])
                        plate_crop = cropped_vehicle[py1:py2, px1:px2]
                        plate_img_path = save_plate_image(plate_crop)

                        ocr_result = reader.readtext(plate_crop)
                        if ocr_result:
                            plate_number = ocr_result[0][-2]

                        # Annotate display
                        cv2.putText(display, f'Plate: {plate_number}', (x1, y1 - 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                    log_to_csv(track_id, speed, plate_number, vehicle_img_path, plate_img_path)
                    print(f"🚨 {label.upper()} ID {track_id} → {speed} km/h → Plate: {plate_number}")

                    cv2.putText(display, f'OVERSPEED {speed} km/h', (x1, y1 - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Show speed above vehicle
            if track_id in vehicle_speeds:
                speed_text = f"{vehicle_speeds[track_id]} km/h"
                color = (0, 0, 255) if vehicle_speeds[track_id] > SPEED_LIMIT_KMPH else (255, 255, 0)
                cv2.putText(display, speed_text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Show live video
        cv2.imshow("Vehicle Speed & Plate Detection", display)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # quit on 'q'
            break

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Processing complete. Logs saved in overspeeding_log.csv")

# ---------------- RUN ----------------
if __name__ == "__main__":
    process_video(VIDEO_PATH)
