import cv2
import mediapipe as mp
import numpy as np
import math

# --- Init FaceMesh ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)


# Utility functions
def euclidean(x1, y1, x2, y2):
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def angle_deg(x1, y1, x2, y2):
    return math.degrees(math.atan2((y2 - y1), (x2 - x1)))


# Extract facial landmarks
def extract_landmarks(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(img_rgb)

    if not result.multi_face_landmarks:
        return None

    lm = result.multi_face_landmarks[0].landmark

    # 10 landmark (x, y) — sama seperti notebook training
    data = {
        "left_eyebrow_x": lm[105].x, "left_eyebrow_y": lm[105].y,
        "right_eyebrow_x": lm[334].x, "right_eyebrow_y": lm[334].y,

        "left_eye_x": lm[33].x, "left_eye_y": lm[33].y,
        "right_eye_x": lm[263].x, "right_eye_y": lm[263].y,

        "nose_x": lm[1].x, "nose_y": lm[1].y,

        "mouth_left_x": lm[61].x, "mouth_left_y": lm[61].y,
        "mouth_right_x": lm[291].x, "mouth_right_y": lm[291].y,
    }

    return data



# Normalization
def normalize_coordinates(data):
    xs = [
        data['left_eyebrow_x'], data['right_eyebrow_x'],
        data['left_eye_x'], data['right_eye_x'],
        data['nose_x'],
        data['mouth_left_x'], data['mouth_right_x']
    ]

    ys = [
        data['left_eyebrow_y'], data['right_eyebrow_y'],
        data['left_eye_y'], data['right_eye_y'],
        data['nose_y'],
        data['mouth_left_y'], data['mouth_right_y']
    ]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    scale_x = max_x - min_x
    scale_y = max_y - min_y

    if scale_x == 0: scale_x = 1e-6
    if scale_y == 0: scale_y = 1e-6

    norm = data.copy()

    # normalisasi persis seperti notebook
    for key in data.keys():
        if key.endswith("_x"):
            norm[key] = (data[key] - min_x) / scale_x
        elif key.endswith("_y"):
            norm[key] = (data[key] - min_y) / scale_y

    return norm



# Feature Engineering — 9 fitur
def compute_features(d):
    fitur = {
        # Asimetri alis
        "delta_eyebrow_y": abs(d['left_eyebrow_y'] - d['right_eyebrow_y']),
        "delta_eyebrow_x": abs(d['left_eyebrow_x'] - d['right_eyebrow_x']),

        # Asimetri mata
        "delta_eye_y": abs(d['left_eye_y'] - d['right_eye_y']),
        "delta_eye_x": abs(d['left_eye_x'] - d['right_eye_x']),

        # Jarak mata–hidung
        "dist_eye_left_to_nose": euclidean(d['left_eye_x'],  d['left_eye_y'],  d['nose_x'], d['nose_y']),
        "dist_eye_right_to_nose": euclidean(d['right_eye_x'], d['right_eye_y'], d['nose_x'], d['nose_y']),

        "delta_eye_nose": abs(
            euclidean(d['left_eye_x'],  d['left_eye_y'],  d['nose_x'], d['nose_y']) -
            euclidean(d['right_eye_x'], d['right_eye_y'], d['nose_x'], d['nose_y'])
        ),

        # Jarak bibir kiri–kanan
        "dist_mouth": euclidean(d['mouth_left_x'], d['mouth_left_y'],
                                d['mouth_right_x'], d['mouth_right_y']),

        # Kemiringan bibir
        "mouth_angle": angle_deg(d['mouth_left_x'], d['mouth_left_y'],
                                 d['mouth_right_x'], d['mouth_right_y']),
    }

    return fitur



# MAIN (dipanggil oleh Streamlit)
def extract_numeric_vector(image_path):
    """
    Output:
        np.array shape (1, 9)
        atau None jika wajah tidak terdeteksi
    """

    data = extract_landmarks(image_path)
    if data is None:
        return None

    norm = normalize_coordinates(data)
    fitur = compute_features(norm)

    vector = np.array([list(fitur.values())])  # (1, 9)
    return vector