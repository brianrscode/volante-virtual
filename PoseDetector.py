import cv2
import math
import numpy as np
import mediapipe as mp
from typing import Tuple, List


class PoseDetector:
    def __init__(
        self,
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        smooth_segmentation=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ):
        self.static_image_mode = static_image_mode
        self.model_complexity = model_complexity
        self.smooth_landmarks = smooth_landmarks
        self.enable_segmentation = enable_segmentation
        self.smooth_segmentation = smooth_segmentation
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        # Objects for hand detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            self.static_image_mode,
            self.model_complexity,
            self.smooth_landmarks,
            self.enable_segmentation,
            self.smooth_segmentation,
            self.min_detection_confidence,
            self.min_tracking_confidence
        )


    def find_pose(self, frame, draw=True) -> np.ndarray:
        '''Hand detection in the frame and drawing of landmarks and connectors'''
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(img_rgb)

        if draw:
            if self.results.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    self.results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2),
                    self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
                )
                    # landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
        return frame


    def find_position_landmarks(self, frame, draw=True) -> List[int]:
        """
        Detecting the position of each landmark and drawing them
        Args:
            frame: the frame in which landmarks are detected
            draw: if True, landmarks are drawn
        Returns:
            landmark_list: list containing each landmark and its position
        """

        # x_list = []
        # y_list = []
        self.landmark_list = []

        if self.results.pose_landmarks:
            for id_landmark, landmark in enumerate(self.results.pose_landmarks.landmark):
                h, w, _ = frame.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)

                # x_list.append(cx)
                # y_list.append(cy)
                self.landmark_list.append([id_landmark, cx, cy])

                if draw:
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 0), cv2.FILLED)

        # print(self.landmark_list)
        return self.landmark_list

    def distance_landmarks(self, p1: int, p2: int, frame: np.ndarray, draw=True, line_width=2) -> Tuple[float, np.ndarray, list[int]]:
        '''Gets the distance between two points (landmarks)'''
        x1, y1 = self.landmark_list[p1][1:]
        x2, y2 = self.landmark_list[p2][1:]

        # Average between two points
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # Distance between two points
        distance = math.hypot(x2 - x1, y2 - y1)
        linea = [x1, y1, x2, y2, cx, cy]

        if draw:
            # Draw a line between the two points
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 0), line_width)
            # Draw the distance between the two points
            # cv2.putText(frame, str(int(distance)), (cx, cy), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)

        return distance, frame, linea

    def distance_points(self, p1: np.array, p2: np.array, frame: np.ndarray, draw_l=True, draw_d=False, color=(0, 0, 0)) -> Tuple[float, np.ndarray]:

        distance = np.linalg.norm(p1 - p2)  # Distancia entre puntos

        if draw_l:
            # Draw a line between the two points
            cv2.line(frame, p1, p2, color, 2)
        if draw_d:
            # Draw the distance between the two points
            x1, y1 = p1
            x2, y2 = p2
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.putText(frame, str(int(distance)), (cx, cy), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)

        return distance, frame
