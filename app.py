import os
import cv2
import numpy as np
import face_recognition
import mediapipe as mp
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import tempfile
import logging

# ─────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mp_face_mesh = mp.solutions.face_mesh

# MediaPipe landmark indices for eyes
LEFT_EYE_TOP    = 159
LEFT_EYE_BOTTOM = 145
LEFT_EYE_LEFT   = 33
LEFT_EYE_RIGHT  = 133

RIGHT_EYE_TOP    = 386
RIGHT_EYE_BOTTOM = 374
RIGHT_EYE_LEFT   = 362
RIGHT_EYE_RIGHT  = 263

BLINK_EAR_THRESHOLD = 0.20   # Eye Aspect Ratio threshold
BLINK_CONSEC_FRAMES = 2       # Min frames eye must be closed
FACE_MATCH_TOLERANCE = 0.50   # Lower = stricter


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _ear(landmarks, top_idx, bottom_idx, left_idx, right_idx):
    """Eye Aspect Ratio for a single eye."""
    top    = landmarks[top_idx]
    bottom = landmarks[bottom_idx]
    left   = landmarks[left_idx]
    right  = landmarks[right_idx]

    vertical   = abs(top.y - bottom.y)
    horizontal = abs(left.x - right.x)
    if horizontal == 0:
        return 0.0
    return vertical / horizontal


def detect_blink(video_path: str) -> bool:
    """Return True if at least one blink is detected in the video."""
    cap = cv2.VideoCapture(video_path)
    blink_detected = False
    consec = 0

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(rgb)

            if not result.multi_face_landmarks:
                consec = 0
                continue

            lm = result.multi_face_landmarks[0].landmark

            left_ear  = _ear(lm, LEFT_EYE_TOP,  LEFT_EYE_BOTTOM,  LEFT_EYE_LEFT,  LEFT_EYE_RIGHT)
            right_ear = _ear(lm, RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM, RIGHT_EYE_LEFT, RIGHT_EYE_RIGHT)
            avg_ear   = (left_ear + right_ear) / 2.0

            if avg_ear < BLINK_EAR_THRESHOLD:
                consec += 1
            else:
                if consec >= BLINK_CONSEC_FRAMES:
                    blink_detected = True
                    break
                consec = 0

    cap.release()
    return blink_detected


def extract_face_encoding_from_video(video_path: str):
    """
    Sample frames from the video, find the first frame with exactly one face,
    and return its 128-d encoding.  Returns None if no suitable frame is found.
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_step  = max(1, total_frames // 20)   # sample ~20 frames
    encoding     = None
    frame_idx    = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_step == 0:
            rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            locs = face_recognition.face_locations(rgb, model="hog")

            if len(locs) == 1:
                encs = face_recognition.face_encodings(rgb, locs)
                if encs:
                    encoding = encs[0]
                    break

        frame_idx += 1

    cap.release()
    return encoding


def _largest_face_index(locs):
    """Return index of the largest face bounding box (top, right, bottom, left)."""
    def area(loc):
        top, right, bottom, left = loc
        return (bottom - top) * (right - left)
    return max(range(len(locs)), key=lambda i: area(locs[i]))


def get_face_encoding_from_image(image_path: str):
    """
    Load an image and return the encoding of the largest detected face.
    If multiple faces are found, the largest (most prominent) is used automatically.
    Returns (None, 'no_face') only when zero faces are detected.
    """
    img  = face_recognition.load_image_file(image_path)
    locs = face_recognition.face_locations(img, model="hog")

    if len(locs) == 0:
        logger.warning("No face detected in %s", image_path)
        return None, "no_face"

    if len(locs) > 1:
        logger.info("Multiple faces in %s — picking largest face", image_path)
        locs = [locs[_largest_face_index(locs)]]

    encs = face_recognition.face_encodings(img, locs)
    return (encs[0] if encs else None), "ok"


def faces_match(enc1, enc2, tolerance: float = FACE_MATCH_TOLERANCE) -> bool:
    """Return True if two encodings are close enough to be the same person."""
    if enc1 is None or enc2 is None:
        return False
    distance = face_recognition.face_distance([enc1], enc2)[0]
    return distance <= tolerance


def save_upload(file_storage) -> str:
    """Save a FileStorage object to a temp file and return its path."""
    suffix = os.path.splitext(secure_filename(file_storage.filename))[1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    file_storage.save(tmp.name)
    return tmp.name


def cleanup(*paths):
    """Delete temporary files."""
    for p in paths:
        try:
            if p and os.path.exists(p):
                os.remove(p)
        except OSError:
            pass


# ─────────────────────────────────────────────
# Unified Verification Endpoint
# ─────────────────────────────────────────────

@app.route("/verify-listing", methods=["POST"])
def verify_listing():
    print("FILES RECEIVED:", request.files)

    video_file = request.files.get("owner_live_video")
    print("VIDEO NAME:", video_file.filename if video_file else "None")

    photo_file = request.files.get("owner_photo")
    print("PHOTO NAME:", photo_file.filename if photo_file else "None")

    room_files = request.files.getlist("room_images")
    print("ROOM IMAGES COUNT:", len(room_files))
    """
    POST /verify-listing
    Form-data fields:
        owner_live_video  – video file (mp4 / avi / mov …)
        owner_photo       – image file (jpg / png …)
        room_images[]     – one or more image files
    """
    # ── Validate inputs ──────────────────────────────────────────────────────
    if "owner_live_video" not in request.files:
        return jsonify({"verified": False, "reason": "owner_live_video is required"}), 400
    if "owner_photo" not in request.files:
        return jsonify({"verified": False, "reason": "owner_photo is required"}), 400

    room_files = request.files.getlist("room_images")
    if not room_files or all(f.filename == "" for f in room_files):
        return jsonify({"verified": False, "reason": "At least one room_image is required"}), 400

    video_path = photo_path = None
    room_paths = []

    try:
        # ── Save uploads ──────────────────────────────────────────────────────
        video_path = save_upload(request.files["owner_live_video"])
        photo_path = save_upload(request.files["owner_photo"])
        room_paths = [save_upload(f) for f in room_files if f.filename != ""]

        # ── Step 1 – Liveness check ───────────────────────────────────────────
        logger.info("Step 1: Liveness detection …")
        liveness_passed = detect_blink(video_path)
        if not liveness_passed:
            return jsonify({
                "verified":        False,
                "liveness_check":  "FAILED – no blink detected",
                "reason":          "Liveness check failed – no blink detected"
            })

        # ── Step 2 – Extract live face encoding from video ────────────────────
        logger.info("Step 2: Extracting live face from video …")
        live_encoding = extract_face_encoding_from_video(video_path)
        if live_encoding is None:
            return jsonify({
                "verified":       False,
                "liveness_check": "PASSED – blink detected",
                "reason":         "Could not extract a face from the live video"
            })

        # ── Step 3 – Match live face with owner photo ─────────────────────────
        logger.info("Step 3: Matching live face with owner photo …")
        owner_encoding, status = get_face_encoding_from_image(photo_path)
        if owner_encoding is None:
            return jsonify({
                "verified":       False,
                "liveness_check": "PASSED – blink detected",
                "reason":         f"Owner photo issue: {status}"
            })

        if not faces_match(live_encoding, owner_encoding):
            return jsonify({
                "verified":       False,
                "liveness_check": "PASSED – blink detected",
                "reason":         "Live person does not match owner photo"
            })

        # ── Step 4 – Match owner photo with every room image ──────────────────
        logger.info("Step 4: Checking owner presence in %d room image(s) …", len(room_paths))
        total_images   = len(room_paths)
        matched_images = 0
        unmatched      = []

        for idx, rp in enumerate(room_paths):
            room_enc, room_status = get_face_encoding_from_image(rp)

            if room_enc is None:
                # No face / multiple faces → cannot confirm owner presence
                logger.info("  Room image %d: %s", idx + 1, room_status)
                unmatched.append(idx + 1)
                continue

            if faces_match(owner_encoding, room_enc):
                matched_images += 1
                logger.info("  Room image %d: MATCH ✓", idx + 1)
            else:
                unmatched.append(idx + 1)
                logger.info("  Room image %d: NO MATCH ✗", idx + 1)

        # ── Decision ──────────────────────────────────────────────────────────
        listing_real = (matched_images == total_images)

        if listing_real:
            return jsonify({
                "verified":       True,
                "liveness_check": "PASSED – blink detected",
                "message":        "Owner detected in all listing images. Listing is REAL.",
                "matched_images": matched_images,
                "total_images":   total_images,
            })
        else:
            return jsonify({
                "verified":              False,
                "liveness_check":        "PASSED – blink detected",
                "message":               "Owner not present in all listing images. Listing is FAKE.",
                "matched_images":        matched_images,
                "total_images":          total_images,
                "unmatched_image_index": unmatched,
            })

    except Exception as exc:
        logger.exception("Unexpected error during verification")
        return jsonify({"verified": False, "reason": f"Internal error: {str(exc)}"}), 500

    finally:
        cleanup(video_path, photo_path, *room_paths)


# ─────────────────────────────────────────────
# Health check
# ─────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5001)
