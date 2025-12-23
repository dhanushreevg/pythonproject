import cv2

# Open webcam (0 = default camera)
cap = cv2.VideoCapture(0)

# Create background subtractor
bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=500,
    varThreshold=50,
    detectShadows=True
)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Apply background subtraction
    fg_mask = bg_subtractor.apply(frame)

    # Remove noise using morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv2.findContours(
        fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    for contour in contours:
        if cv2.contourArea(contour) < 800:
            continue  # Ignore small movements/noise

        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "Moving Object", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show output
    cv2.imshow("Frame", frame)
    cv2.imshow("Foreground Mask", fg_mask)

    # Press ESC to exit
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
