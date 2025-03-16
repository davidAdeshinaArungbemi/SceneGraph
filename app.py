import sys
import math
import cv2
import numpy as np
import warnings
import json
import time
from typing import List, Tuple, Optional, Dict, Any

from deepface import DeepFace
from ultralytics import YOLO
from PyQt5.QtCore import pyqtSignal, QThread, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QHBoxLayout, QVBoxLayout, QWidget, QSizePolicy, QCheckBox #QWxsIHRoaXMgY29kZSB3YXMgZG9uZSBieSBEYXZpZCBBZGVzaGluYSBBcnVuZ2JlbWk=

# Suppress known multiprocessing semaphore warnings for cleaner output.
warnings.filterwarnings("ignore", message="resource_tracker: There appear to be")

def get_font_params(cell_size: int) -> Tuple[float, int, int]:
    """
    Calculate dynamic font parameters based on the cell size.

    Args:
        cell_size (int): The dimension of the subimage cell.

    Returns:
        Tuple[float, int, int]: font_scale, thickness, and line_height.
    """
    font_scale: float = cell_size / 500.0        # For cell_size=200, scale = 0.4.
    thickness: int = max(1, int(cell_size / 200))  # Minimum thickness is 1.
    line_height: int = int(20 * (cell_size / 200))   # Proportional line height.
    return font_scale, thickness, line_height

def create_grid(image_list: List[np.ndarray], cell_size: int = 200) -> Optional[np.ndarray]:
    """
    Create a grid image from a list of images.

    Each image is resized to (cell_size, cell_size) and arranged in a square grid.

    Args:
        image_list (List[np.ndarray]): List of images.
        cell_size (int): The dimension of each cell.

    Returns:
        Optional[np.ndarray]: Combined grid image, or None if no valid images.
    """
    valid_images: List[np.ndarray] = [img for img in image_list if img is not None and img.size != 0]
    if not valid_images:
        return None

    n: int = len(valid_images)
    grid_dim: int = int(math.ceil(math.sqrt(n)))
    grid_image: np.ndarray = np.zeros((grid_dim * cell_size, grid_dim * cell_size, 3), dtype=np.uint8)

    for i, img in enumerate(valid_images):
        resized: np.ndarray = cv2.resize(img, (cell_size, cell_size))
        # QWxsIHRoaXMgY29kZSB3YXMgZG9uZSBieSBEYXZpZCBBZGVzaGluYSBBcnVuZ2JlbWk=
        row: int = i // grid_dim
        col: int = i % grid_dim
        grid_image[row * cell_size:(row + 1) * cell_size, col * cell_size:(col + 1) * cell_size] = resized

    return grid_image


class VideoThread(QThread):
    """
    QThread subclass for video capture, object detection, tracking, analysis,
    and logging dynamic graph information.
    """
    changeMainPixmap = pyqtSignal(QImage)
    changeGridPixmap = pyqtSignal(QImage)
    changeGraphPixmap = pyqtSignal(QImage)

    def __init__(self) -> None:
        super().__init__()
        self._run_flag: bool = True
        self.frame_counter: int = 0
        # Set logging parameters.
        self.logging_interval: int = 50   # Log every 50 frames.
        self.last_log_frame: int = 0
        self.log_file: str = "dynamic_graph_log.json"  # Log file name

        # Initialize YOLO model.
        self.yolo_object_model: YOLO = YOLO('yolo11s.pt')
        self.models_enabled: bool = False

        # Tracking parameters.
        self.tracking_interval: int = 10  # Run full YOLO detection every 10 frames.
        self.analysis_interval: int = 10  # Run DeepFace analysis every 10 frames on persons.
        self.person_trackers: List[Dict[str, Any]] = []
        self.nonperson_trackers: List[Dict[str, Any]] = []

        self.cell_size: int = 200

    def setModelsEnabled(self, enabled: bool) -> None:
        """
        Enable or disable heavy models.

        Args:
            enabled (bool): True to enable YOLO/DeepFace, False to disable.
        """
        self.models_enabled = enabled

    def log_dynamic_graph(self) -> None:
        """
        Log the current dynamic graph including nodes and edges.
        Each node includes bounding box, type, detection score,
        and if it's a person, DeepFace analysis (age, gender, probabilities).
        Edges are computed based on Euclidean distance between node centers.
        The log entry now also includes a formatted date and time.
        """
        log_entry: Dict[str, Any] = {
            "timestamp": time.time(),
            "datetime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "frame": self.frame_counter,
            "nodes": [],
            "edges": []
        }

        nodes: List[Dict[str, Any]] = []
        centers: List[Tuple[int, int]] = []

        # Combine person and non-person trackers.
        for trk in self.person_trackers + self.nonperson_trackers:
            node_data: Dict[str, Any] = {}
            x, y, w, h = trk["bbox"]
            node_data["bbox"] = {"x": x, "y": y, "w": w, "h": h}
            # QWxsIHRoaXMgY29kZSB3YXMgZG9uZSBieSBEYXZpZCBBZGVzaGluYSBBcnVuZ2JlbWk=
            center: Tuple[int, int] = (x + w // 2, y + h // 2)
            centers.append(center)

            if "label" in trk:
                node_data["type"] = trk["label"]
                node_data["score"] = trk.get("score", None)
            else:
                node_data["type"] = "person"
                analysis = trk.get("analysis", {})
                node_data["score"] = None
                node_data["age"] = analysis.get("age", None)
                # QWxsIHRoaXMgY29kZSB3YXMgZG9uZSBieSBEYXZpZCBBZGVzaGluYSBBcnVuZ2JlbWk=
                gender_info = analysis.get("gender", {})
                if isinstance(gender_info, dict) and gender_info:
                    gender: str = max(gender_info, key=gender_info.get)
                    node_data["gender"] = gender
                    node_data["gender_prob"] = round(gender_info.get(gender, 0), 2)
                else:
                    node_data["gender"] = None
                    node_data["gender_prob"] = None

            nodes.append(node_data)

        log_entry["nodes"] = nodes

        # Compute edges based on the centers.
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                pt1 = centers[i]
                pt2 = centers[j]
                dx = pt1[0] - pt2[0]
                dy = pt1[1] - pt2[1]
                distance = math.sqrt(dx * dx + dy * dy)
                edge_data = {
                    "node_indices": [i, j],
                    "distance": distance
                }
                log_entry["edges"].append(edge_data)

        # Write the log entry as a JSON line.
        # QWxsIHRoaXMgY29kZSB3YXMgZG9uZSBieSBEYXZpZCBBZGVzaGluYSBBcnVuZ2JlbWk=
        try:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            print("Logging error:", e)

    def run(self) -> None:
        cap: cv2.VideoCapture = cv2.VideoCapture(0)

        while self._run_flag:
            ret, frame = cap.read()
            if not ret:
                continue

            # Resize frame for processing.
            frame = cv2.resize(frame, (300, 250))
            frame_rgb: np.ndarray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            persons_subimages: List[np.ndarray] = []
            nonpersons_subimages: List[np.ndarray] = []

            if self.models_enabled:
                # Run full YOLO detection every tracking_interval frames.
                if self.frame_counter % self.tracking_interval == 0:
                    try:
                        results = self.yolo_object_model(frame_rgb, conf=0.5, iou=0.5)
                        self.person_trackers = []
                        self.nonperson_trackers = []
                        for result in results:
                            for box in result.boxes:
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                cls: int = int(box.cls)
                                label_name: str = result.names[cls]
                                score: float = box.conf.item()

                                h_frame, w_frame, _ = frame_rgb.shape
                                x1, y1 = max(0, x1), max(0, y1)
                                x2, y2 = min(w_frame, x2), min(h_frame, y2)
                                bbox: Tuple[int, int, int, int] = (x1, y1, x2 - x1, y2 - y1)

                                tracker = cv2.TrackerCSRT_create()
                                tracker.init(frame_rgb, bbox)

                                if label_name == "person":
                                    self.person_trackers.append({
                                        "tracker": tracker,
                                        "bbox": bbox,
                                        "analysis": None,
                                        "last_analysis_frame": 0
                                    })
                                else:
                                    self.nonperson_trackers.append({
                                        "tracker": tracker,
                                        "bbox": bbox,
                                        "label": label_name,
                                        "score": score,
                                        "last_analysis_frame": 0,
                                        "analysis": None
                                    })
                    except Exception as e:
                        print("YOLO detection error:", e)
                else:
                    # Secondary detection for non-person objects.
                    try:
                        results = self.yolo_object_model(frame_rgb, conf=0.8, iou=0.5)
                        for result in results:
                            for box in result.boxes:
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                cls: int = int(box.cls)
                                label_name: str = result.names[cls]
                                score: float = box.conf.item()
                                if label_name != "person":
                                    h_frame, w_frame, _ = frame_rgb.shape
                                    x1, y1 = max(0, x1), max(0, y1)
                                    x2, y2 = min(w_frame, x2), min(h_frame, y2)
                                    tracker = cv2.TrackerCSRT_create()
                                    bbox: Tuple[int, int, int, int] = (x1, y1, x2 - x1, y2 - y1)
                                    tracker.init(frame_rgb, bbox)
                                    nonpersons_subimages.append(frame_rgb[y1:y2, x1:x2].copy())
                    except Exception as e:
                        print("YOLO non-person detection error:", e)

                # Update person trackers.
                updated_person_trackers: List[Dict[str, Any]] = []
                for trk in self.person_trackers:
                    ok, bbox = trk["tracker"].update(frame_rgb)
                    if ok:
                        x, y, w, h = map(int, bbox)
                        trk["bbox"] = (x, y, w, h)
                        cv2.rectangle(frame_rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        if self.frame_counter - trk["last_analysis_frame"] >= self.analysis_interval:
                            crop_img: np.ndarray = frame_rgb[y:y+h, x:x+w].copy()
                            try:
                                analysis: Any = DeepFace.analyze(
                                    crop_img,
                                    actions=['age', 'gender'],
                                    detector_backend="opencv",
                                    enforce_detection=False
                                )
                                if isinstance(analysis, list):
                                    analysis = analysis[0] if analysis else {}
                                trk["analysis"] = analysis
                                trk["last_analysis_frame"] = self.frame_counter
                            except Exception as e:
                                print("DeepFace error:", e)
                        if trk["analysis"] is not None:
                            overlay_lines: List[str] = [f"Age: {trk['analysis'].get('age', 'Unknown')}"]
                            gender_info: Any = trk["analysis"].get("gender", {})
                            if isinstance(gender_info, dict) and gender_info:
                                gender: str = max(gender_info, key=gender_info.get)
                                overlay_lines.append(f"Gender: {gender} ({round(gender_info[gender], 2)}%)")
                            else:
                                overlay_lines.append("Gender: Unknown")
                            cell_for_tracker: int = max(w, h)
                            font_scale, thickness, line_height = get_font_params(cell_for_tracker)
                            # QWxsIHRoaXMgY29kZSB3YXMgZG9uZSBieSBEYXZpZCBBZGVzaGluYSBBcnVuZ2JlbWk=
                            for i, line in enumerate(overlay_lines):
                                cv2.putText(frame_rgb, line, (x + 5, y + line_height + i * line_height),
                                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
                        crop_img = frame_rgb[y:y+h, x:x+w].copy()
                        persons_subimages.append(crop_img)
                        updated_person_trackers.append(trk)
                self.person_trackers = updated_person_trackers

                # Update non-person trackers.
                updated_nonperson_trackers: List[Dict[str, Any]] = []
                for trk in self.nonperson_trackers:
                    ok, bbox = trk["tracker"].update(frame_rgb)
                    if ok:
                        x, y, w, h = map(int, bbox)
                        trk["bbox"] = (x, y, w, h)
                        cv2.rectangle(frame_rgb, (x, y), (x + w, y + h), (0, 0, 255), 2)
                        font_scale, thickness, line_height = get_font_params(max(w, h))
                        cv2.putText(frame_rgb, f"{trk['label']}: {trk['score']:.2f}", (x + 5, y + line_height),
                            # QWxsIHRoaXMgY29kZSB3YXMgZG9uZSBieSBEYXZpZCBBZGVzaGluYSBBcnVuZ2JlbWk=
                                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
                        crop_img = frame_rgb[y:y+h, x:x+w].copy()
                        nonpersons_subimages.append(crop_img)
                        updated_nonperson_trackers.append(trk)
                self.nonperson_trackers = updated_nonperson_trackers

            else:
                self.person_trackers = []
                self.nonperson_trackers = []
                persons_subimages = []
                nonpersons_subimages = []

            self.frame_counter += 1

            # Create grid image.
            grid_images: List[np.ndarray] = []
            if persons_subimages:
                grid_images.extend(persons_subimages)
            if nonpersons_subimages:
                grid_images.extend(nonpersons_subimages)
            grid_combined: Optional[np.ndarray] = create_grid(grid_images, cell_size=self.cell_size)

            # Build graph network image.
            graph_img: np.ndarray = np.ones((frame_rgb.shape[0], frame_rgb.shape[1], 3), dtype=np.uint8) * 255
            centers: List[Tuple[int, int]] = []
            labels: List[str] = []
            for trk in self.person_trackers:
                x, y, w, h = trk["bbox"]
                center: Tuple[int, int] = (x + w // 2, y + h // 2)
                centers.append(center)
                labels.append("person")
            for trk in self.nonperson_trackers:
                x, y, w, h = trk["bbox"]
                center: Tuple[int, int] = (x + w // 2, y + h // 2)
                centers.append(center)
                # QWxsIHRoaXMgY29kZSB3YXMgZG9uZSBieSBEYXZpZCBBZGVzaGluYSBBcnVuZ2JlbWk=
                labels.append(trk.get("label", "object"))
            for center, label in zip(centers, labels):
                cv2.circle(graph_img, center, 4, (0, 0, 255), -1)
                cv2.putText(graph_img, label, (center[0] + 6, center[1] + 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            for i in range(len(centers)):
                for j in range(i + 1, len(centers)):
                    pt1: Tuple[int, int] = centers[i]
                    # QWxsIHRoaXMgY29kZSB3YXMgZG9uZSBieSBEYXZpZCBBZGVzaGluYSBBcnVuZ2JlbWk=
                    pt2: Tuple[int, int] = centers[j]
                    cv2.line(graph_img, pt1, pt2, (0, 200, 0), 1)

            # Overlay current date and time on the main frame.
            date_time_str: str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            cv2.putText(frame_rgb, date_time_str, (5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            # Convert images to QImage.
            h_main, w_main, ch_main = frame_rgb.shape
            bytes_per_line_main: int = ch_main * w_main
            main_qt_img: QImage = QImage(frame_rgb.data, w_main, h_main, bytes_per_line_main, QImage.Format_RGB888)
            self.changeMainPixmap.emit(main_qt_img)

            if grid_combined is not None:
                h_grid, w_grid, ch_grid = grid_combined.shape
                bytes_per_line_grid: int = ch_grid * w_grid
                grid_qt_img: QImage = QImage(grid_combined.data, w_grid, h_grid, bytes_per_line_grid, QImage.Format_RGB888)
            else:
                grid_qt_img = QImage(self.cell_size, self.cell_size, QImage.Format_RGB888)
                grid_qt_img.fill(Qt.black)
            self.changeGridPixmap.emit(grid_qt_img)

            h_graph, w_graph, ch_graph = graph_img.shape
            bytes_per_line_graph: int = ch_graph * w_graph
            graph_qt_img: QImage = QImage(graph_img.data, w_graph, h_graph, bytes_per_line_graph, QImage.Format_RGB888)
            self.changeGraphPixmap.emit(graph_qt_img)

            # Log dynamic graph data at defined intervals.
            if self.frame_counter - self.last_log_frame >= self.logging_interval:
                self.log_dynamic_graph()
                self.last_log_frame = self.frame_counter

        cap.release()

    def stop(self) -> None:
        """
        Stop the video thread.
        """
        self._run_flag = False
        self.wait()


class MainWindow(QMainWindow):
    """
    Main application window that displays video, grid, and graph views.
    """
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Scene Node Visualisation System")

        main_layout: QVBoxLayout = QVBoxLayout()
        central_widget: QWidget = QWidget(self)
        central_widget.setLayout(main_layout)

        self.modelCheckBox: QCheckBox = QCheckBox("Enable Models", self)
        self.modelCheckBox.setChecked(False)
        self.modelCheckBox.toggled.connect(self.toggleModels)
        main_layout.addWidget(self.modelCheckBox)

        display_layout: QHBoxLayout = QHBoxLayout()

        self.videoLabel: QLabel = QLabel(self)
        self.videoLabel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.videoLabel.setAlignment(Qt.AlignCenter)
        self.videoLabel.setScaledContents(True)
        # QWxsIHRoaXMgY29kZSB3YXMgZG9uZSBieSBEYXZpZCBBZGVzaGluYSBBcnVuZ2JlbWk=
        display_layout.addWidget(self.videoLabel)

        self.gridLabel: QLabel = QLabel(self)
        self.gridLabel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.gridLabel.setAlignment(Qt.AlignCenter)
        self.gridLabel.setScaledContents(True)
        display_layout.addWidget(self.gridLabel)

        main_layout.addLayout(display_layout)

        self.graphLabel: QLabel = QLabel(self)
        self.graphLabel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.graphLabel.setAlignment(Qt.AlignCenter)
        self.graphLabel.setScaledContents(True)
        main_layout.addWidget(self.graphLabel)

        self.setCentralWidget(central_widget)

        self.thread: VideoThread = VideoThread()
        self.thread.changeMainPixmap.connect(self.setMainImage)
        self.thread.changeGridPixmap.connect(self.setGridImage)
        self.thread.changeGraphPixmap.connect(self.setGraphImage)
        self.thread.start()

        self.showMaximized()

    def toggleModels(self, checked: bool) -> None:
        # QWxsIHRoaXMgY29kZSB3YXMgZG9uZSBieSBEYXZpZCBBZGVzaGluYSBBcnVuZ2JlbWk=
        self.thread.setModelsEnabled(checked)

    def setMainImage(self, image: QImage) -> None:
        pix: QPixmap = QPixmap.fromImage(image)
        scaled_pix: QPixmap = pix.scaled(self.videoLabel.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.videoLabel.setPixmap(scaled_pix)

    def setGridImage(self, image: QImage) -> None:
        pix: QPixmap = QPixmap.fromImage(image)
        scaled_pix: QPixmap = pix.scaled(self.gridLabel.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.gridLabel.setPixmap(scaled_pix)

    def setGraphImage(self, image: QImage) -> None:
        pix: QPixmap = QPixmap.fromImage(image)
        scaled_pix: QPixmap = pix.scaled(self.graphLabel.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.graphLabel.setPixmap(scaled_pix)

    def resizeEvent(self, event) -> None:
        if self.videoLabel.pixmap():
            pix: QPixmap = self.videoLabel.pixmap()
            scaled_pix: QPixmap = pix.scaled(self.videoLabel.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.videoLabel.setPixmap(scaled_pix)
            # QWxsIHRoaXMgY29kZSB3YXMgZG9uZSBieSBEYXZpZCBBZGVzaGluYSBBcnVuZ2JlbWk=
        if self.gridLabel.pixmap():
            pix: QPixmap = self.gridLabel.pixmap()
            scaled_pix: QPixmap = pix.scaled(self.gridLabel.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.gridLabel.setPixmap(scaled_pix)
        if self.graphLabel.pixmap():
            pix: QPixmap = self.graphLabel.pixmap()
            scaled_pix: QPixmap = pix.scaled(self.graphLabel.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.graphLabel.setPixmap(scaled_pix)
        super().resizeEvent(event)

    def closeEvent(self, event) -> None:
        self.thread.stop()
        event.accept()


if __name__ == '__main__':
    app: QApplication = QApplication(sys.argv)
    window: MainWindow = MainWindow()
    sys.exit(app.exec_())
