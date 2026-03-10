import cv2
from cv2.typing import MatLike
from ultralytics import YOLO
import torch
from torch import cuda, device
import torch.nn as nn
from torchvision.models import EfficientNet, efficientnet_b0, EfficientNet_B0_Weights
from torchvision.transforms import v2
from google import genai
from dotenv import load_dotenv
from typing import Optional
from datetime import datetime
import os
import time


DEVICE: device = "cuda" if cuda.is_available() else "cpu"
FACIAL_DETECTIOM_FILEPATH = "./yolov8n-face-lindevs.pt"
AVAILABLE_EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
NUM_TOP_EMOTIONS = 3
TEXT_COLOR = cv2.FONT_HERSHEY_SIMPLEX
CONVERSATION_MODE = False

# TODO: Train the efficientnet and then put the params file here


class CVPipeline:
    def __init__(self, face_detection_params: str, device: device) -> None:
        """
        The CV pipeline that handles camera and detect emotions
        """
        self.face_detector = self._init_face_detector(face_detection_params, device)
        self.emotion_classifier = self._init_emotion_detector(device)
        self.emotion_classifier.eval()
        self.camera: Optional[cv2.VideoCapture] = None
    

    def _init_face_detector(self, filepath: str, device) -> YOLO:
        """Helper method: Get the YOLO-based face detection model"""
        detector = YOLO(filepath).to(device)
        return detector


    def _init_emotion_detector(self, device: device) -> EfficientNet:
        """Helper method: Initialize the emotional classification model"""
        efficient = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

        # Modify the linear head
        efficient.classifier[1] = nn.Sequential(
            nn.Linear(efficient.classifier[1].in_features, 4098, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4098, 1024, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 7, bias=True)
        )

        # Freeze every layer of the model except chosen one
        trainable_layers = ["classifier"]
        for name, param in efficient.named_parameters():
            trainable = False
            for trainable_layer in trainable_layers:
                if trainable_layer in name:
                    trainable = True
                    break

            param.requires_grad = trainable
        return efficient.to(device)


    def _convert_to_tensor(self, image: MatLike, device: device) -> torch.Tensor:
        """Helper method: Convert the cv2 image to torch tensor"""
        gray = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        # Scale, normalize the tensorized img
        transform = v2.Compose([
            v2.Resize(224),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        return transform(gray).unsqueeze(0).to(device)


    def get_camera(self) -> Optional[cv2.VideoCapture]:
        """Get the camera (OOP 101)"""
        return self.camera


    def turn_on_camera(self) -> None:
        """Attempt to turn on the camera"""
        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            raise ValueError("Failed to open the camera")
        for _ in range(7):
            # Warm up the camera to prevent it from crashing when reopening
            self.camera.read()
        print("Turned on camera successfully!\n")


    def turn_off_camera(self) -> None:
        """Turn off the camera is any"""
        if self.camera is not None:
            self.camera.release()
            cv2.destroyAllWindows()
            print("Turned off the camera successfully!\n")
            self.camera = None


    def run(self) -> list:
        """
        Detects the face and classifies emotion. Potentially use ArcFace to recognize the face.
        """
        emotions = []
        if not self.camera: return emotions
        success, image = self.camera.read()
        if not success:
            raise ValueError("Failed to capture the image of the user")

        # The face detector captures only one image at the time
        analysis = self.face_detector(image)[0]
        boxes = analysis.boxes.xyxy.cpu().numpy()

        if boxes is not None:
            x1, y1, x2, y2 = map(int, boxes[0])
            face_region: MatLike = image[y1:y2, x1:x2]

            # Feed to emotion classifier
            output = self.emotion_classifier(self._convert_to_tensor(face_region, DEVICE))
            _, labels = torch.topk(output, NUM_TOP_EMOTIONS)
            emotions = [AVAILABLE_EMOTIONS[l] for l in labels[0].cpu().numpy()]
                    
            try:
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(image, ", ".join(emotions), (10, 30), TEXT_COLOR, 0.9, (0, 255, 0), 2)
                timestamp = datetime.now().strftime("%H:%M:%S")
                cv2.imwrite(f"./images/face_{timestamp}.jpg", image)
            except Exception as e:
                raise Exception("Didn't save the img for some reason", e)

        print(f"Predicted range of emotions: {emotions}")
        return emotions


def conversation(llm, cv_pipeline: CVPipeline) -> None:
    """Flow of the conversation, returns the updated camera state"""
    global CONVERSATION_MODE

    def _get_system_prompt(e: list[str]) -> str:
        """System prompt. Feel free to tweak as much as you want"""
        return (
            f"Predicted range of emotions of user: {", ".join(e)} sorted by dominane."
            "You should talk appropriately based on user's emotions."
            "Talk directly to the user, don't give robotic analysis on their emotions."
            "If the user says bye, hope they all the best, and end conversation."
            "The prompt: "
        )

    user_prompt: str = input("You: ")
    if "hello" in user_prompt.lower() and not cv_pipeline.get_camera():
        CONVERSATION_MODE = True
        print("Robot is in conversational mode")
        cv_pipeline.turn_on_camera()

    if CONVERSATION_MODE:
        emotions = cv_pipeline.run()
        response = llm.models.generate_content(
            model="gemini-2.0-flash-lite", 
            contents=_get_system_prompt(emotions) + user_prompt
        )
        print(f"Robot: {response.text}")

    if "bye" in user_prompt.lower() and cv_pipeline.get_camera():
        CONVERSATION_MODE = False
        print("Robot is in sleeping mode")
        cv_pipeline.turn_off_camera()


if __name__ == "__main__":
    try:
        load_dotenv()
        llm = genai.Client()
        cv_pipeline = CVPipeline(FACIAL_DETECTIOM_FILEPATH, DEVICE)

        # Create the images folder if it doesn't exist yet
        os.makedirs("./images", exist_ok=True)
    except Exception as e:
        raise Exception(f"Error occured when initializing pipelines\n{e}")

    while True:
        conversation(llm, cv_pipeline)
        time.sleep(1)