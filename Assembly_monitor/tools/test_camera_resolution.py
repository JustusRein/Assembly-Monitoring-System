import cv2

# Simple Logitech resolution capability test.
# Does NOT modify your main monitoring settings.

CAMERA_INDEX = 0
REQUESTED_WIDTH = 1280
REQUESTED_HEIGHT = 720
REQUESTED_FPS = 30

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("ERROR: Could not open Logitech camera.")
    raise SystemExit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, REQUESTED_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, REQUESTED_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, REQUESTED_FPS)

actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
actual_fps = cap.get(cv2.CAP_PROP_FPS)

print("=" * 60)
print("LOGITECH CAMERA RESOLUTION TEST")
print("=" * 60)
print(f"Requested : {REQUESTED_WIDTH} x {REQUESTED_HEIGHT} @ {REQUESTED_FPS} fps")
print(f"Actual    : {actual_width} x {actual_height} @ {actual_fps:.1f} fps")
print("=" * 60)
print("Press Q in the camera window to quit.")

while True:
    ok, frame = cap.read()

    if not ok:
        print("ERROR: Camera frame read failed.")
        break

    cv2.putText(
        frame,
        f"Actual: {actual_width}x{actual_height}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    cv2.imshow("Logitech Resolution Test", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
