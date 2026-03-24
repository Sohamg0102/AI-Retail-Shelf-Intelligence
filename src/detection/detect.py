from ultralytics import YOLO

def run_detection(image_path):
    """
    Runs object detection on a given image
    """

    # Load pretrained YOLO model
    model = YOLO("yolov8n.pt")

    # Run inference
    results = model(image_path, show=True)

    # Return results for further processing
    return results


if __name__ == "__main__":
    # Test image (temporary)
    image = "https://ultralytics.com/images/bus.jpg"
    
    results = run_detection(image)

    # Print detection output
    for r in results:
        print(r.boxes)