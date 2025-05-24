from ultralytics import YOLO
import cv2
import numpy as np
from kitti360scripts import viewer
# Load YOLOv11x segmentation model
model = YOLO("yolo11x-seg.pt")

# Load image
image_path = "KITTI-360_sample/data_2d_raw/2013_05_28_drive_0000_sync/image_00/data_rect/0000000250.png"
image = cv2.imread(image_path)





# Run inference
results = model(image)[0]

# Visualize segmentation results
for i in range(len(results.masks.data)):
    class_id = int(results.boxes.cls[i].item())
    class_name = model.names[class_id]

    if class_name == "car":
        mask = results.masks.data[i].cpu().numpy()

        # Resize magitsk to match image
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

        # Apply a color to the mask
        color_mask = np.zeros_like(image)
        color_mask[:, :, 0] = mask * 255  # Blue
        overlay = cv2.addWeighted(image, 1.0, color_mask, 0.5, 0)

        # Draw bounding box
        box = results.boxes.xyxy[i].cpu().numpy().astype(int)
        cv2.rectangle(overlay, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
        cv2.putText(overlay, class_name, (box[0], box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        image = overlay

# Show result
cv2.imshow("Car Segmentation", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

viewer("KITTI-360_sample/data")
