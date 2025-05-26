
"""
A Pan Tilt Zoom (PTZ) boat tracking system that detects boats with a YOLO
model, keeps the target centred via PID controlled servos, and applies
digital zoom.

classes:
    - :class:`Config` : all runtime constants
    - :class:`PID` : incremental PID controller
    - :class:`ServoController` : serial I/O to Arduino
    - :class:`BBoxSmoother` : moving average of bbox centre and width
    - :class:`ZoomController` : digital zoom logic
    - :class:`PTZTracker` : main application logic


Usage
Run the module directly:
>>> python3 boat_tracking_servo.py
Press **q** to quit the preview window.


"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Final, Optional, Tuple

import cv2  
import numpy as np  
import serial 
from ultralytics import YOLO  

# Constants

@dataclass(frozen=True)
class Config:
    """ Configuration constants for the PTZ tracker.
    """
    # ─── Serial port and model path ──────────────────────────────────────────────
    SERIAL_PORT: str = "/dev/ttyUSB0" #  Arduino serial port
    BAUDRATE: int = 115200
    MODEL_PATH: Path = Path("models/best.pt")  # Path to the YOLO model
    CONF_THRESH: float = 0.50

    # ─── PID gains ──────────────────────────────────────────────────────────
    KP_PAN: float = 0.015
    KI_PAN: float = 0.000
    KD_PAN: float = 0.003
    KP_TILT: float = 0.015
    KI_TILT: float = 0.000
    KD_TILT: float = 0.003

    # ─── Behaviour and timing ─────────────────────────────────────────────────
    DEAD_ZONE_PX: int = 60  # ignore tiny errors inside this radius
    MAX_STEP_DEG: float = 2.0  # servo step clamp per control cycle
    SEND_INTERVAL: float = 0.05  # seconds between serial writes

    # ─── Servo limits (degrees) ─────────────────────────────────────────────────
    PAN_MIN: int = 0
    PAN_MAX: int = 180
    TILT_MIN: int = 0
    TILT_MAX: int = 180

    # ─── Digital zoom parameters ────────────────────────────────────────────
    MAX_ZOOM: float = 2.5  # maximum zoom factor
    ZOOM_IN_FRAC: float = 0.23  # Zoom‑IN threshold (width ratio)
    ZOOM_OUT_FRAC: float = 0.34  # Zoom‑OUT threshold 
    ZOOM_DAMP: float = 0.10  # low pass coefficient for zoom level

    # ─── Bounding box smoothing ─────────────────────────────────────────────
    SMOOTH_WINDOW: int = 5  # frames for moving average

# Helper classes

class PID:
    """Incremental PID controller
    """

    def __init__(self, kp: float, ki: float, kd: float, *, setpoint: float = 0.0) -> None:
        self.kp: Final = kp
        self.ki: Final = ki
        self.kd: Final = kd
        self.setpoint: float = setpoint
        self.prev_err: float = 0.0
        self.integral: float = 0.0
        self.prev_time: float = time.time()

    def update(self, measurement: float) -> float:
        """Return the output control for the given measurements.
        Args:
            measurement (float): The current measurement.

        Returns:
            float: The output effort.
        
        """
        now = time.time()
        dt = max(now - self.prev_time, 1e-3)  # prevent zero division
        err = self.setpoint - measurement
        self.integral += err * dt
        deriative = (err - self.prev_err) / dt
        self.prev_err, self.prev_time = err, now
        return self.kp * err + self.ki * self.integral + self.kd * deriative


class ServoController:
    """Handles serial I/O to the Arduino servo driver."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.serial = serial.Serial(cfg.SERIAL_PORT, cfg.BAUDRATE, timeout=1)
        time.sleep(2)  # Allow Arduino reset
        self.pan: int = 120  # default for my setup
        self.tilt: int = 10
        self.last_send: float = 0.0

    
    def update(self, d_pan: float, d_tilt: float) -> None:
        """Update servo angles with clamped deltas.

        Args:
            d_pan (float): Change in pan angle.
            d_tilt (float): Change in tilt angle.

        Returns:
            None
        
        """
        cfg =self.cfg
        self.pan = int(np.clip(self.pan + d_pan, cfg.PAN_MIN, cfg.PAN_MAX))
        self.tilt = int(np.clip(self.tilt + d_tilt, cfg.TILT_MIN, cfg.TILT_MAX))

        now = time.time()
        if now - self.last_send >= cfg.SEND_INTERVAL:
            self.serial.write(f"{self.pan},{self.tilt}\n".encode())
            self.last_send = now

    
    def close(self) -> None: 
        """Close the serial port. 
        """
        self.serial.close()


class BBoxSmoother:
    """Sliding window moving average for bbox centre and width."""

    def __init__(self, window: int) -> None:
        # deques of fixed length
        self._dq_x: Deque[float] = deque(maxlen=window)
        self._dq_y: Deque[float] = deque(maxlen=window)
        self._dq_w: Deque[float] = deque(maxlen=window)

    
    def push(self, centre_x: float, centre_y: float, width: float) -> None:
        self._dq_x.append(centre_x)
        self._dq_y.append(centre_y)
        self._dq_w.append(width)

    
    def mean(self) -> Tuple[float, float, float]:
        """Return x, y, width averages.
        Returns:
            Tuple[float, float, float]: The averages of x, y, and width.
        
        """
        def _m(dq: Deque[float]) -> float:
            return float(sum(dq) / len(dq)) if dq else 0.0

        return _m(self._dq_x), _m(self._dq_y), _m(self._dq_w)


class ZoomController:
    """Digital zoom logic for the PTZ tracker."""

    def __init__(self, cfg: Config) -> None:
       self.cfg = cfg
       self.zoom: float = 1.0

    
    @property
    def level(self) -> float:
        """Return the current zoom level."""
        return self.zoom

    
    def update(self, bbox_width: float, frame_width: int) -> None:
        """Update internal zoom according to bbox width relative to frame width.
        Args:
            bbox_width (float): Width of the detected bounding box.
            frame_width (int): Width of the video frame.

        Returns:
            None
        
        """ 
        cfg =self.cfg
        frac = bbox_width / frame_width if frame_width else 1.0
        target = self.zoom  # default stickiness
        if frac < cfg.ZOOM_IN_FRAC:
            target = min(cfg.MAX_ZOOM, max(1.0 / frac, 1.0))
        elif frac > cfg.ZOOM_OUT_FRAC:
            target = 1.0
        # Apply low pass filter to zoom level
        self.zoom += (target - self.zoom) * cfg.ZOOM_DAMP

    
    def relax(self) -> None:
        """Slowly return to 1x when the target is lost.
        """
        self.zoom += (1.0 - self.zoom) *self.cfg.ZOOM_DAMP

    
    @staticmethod
    def apply(frame: np.ndarray, zoom: float) -> np.ndarray:  
        """Return frame with digital zoom applied.
        Args:
            frame (np.ndarray): The input frame.
            zoom (float): The zoom level.

        Returns:
            np.ndarray: The zoomed frame.
        
        """
        if zoom <= 1.001:
            return frame
        h, w = frame.shape[:2]
        new_w, new_h = int(w / zoom), int(h / zoom)
        cx, cy = w // 2, h // 2
        x1, y1 = max(cx - new_w // 2, 0), max(cy - new_h // 2, 0)
        cropped = frame[y1 : y1 + new_h, x1 : x1 + new_w]
        return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)




class PTZTracker:
    """Main application logic for the PTZ boat tracker."""

    def __init__(self, cfg: Optional[Config] = None) -> None:
        self.cfg = cfg or Config()
        self.servo = ServoController(self.cfg)
        self.smoother = BBoxSmoother(self.cfg.SMOOTH_WINDOW)
        self.zoom_ctl = ZoomController(self.cfg)
        self.pid_pan = PID(self.cfg.KP_PAN, self.cfg.KI_PAN, self.cfg.KD_PAN)
        self.pid_tilt = PID(self.cfg.KP_TILT, self.cfg.KI_TILT, self.cfg.KD_TILT)

        # YOLO model
        self.model = YOLO(str(self.cfg.MODEL_PATH))
        device = "cuda" if cv2.cuda.getCudaEnabledDeviceCount() else "cpu"
        self.model.to(device)

        # Camera setup
        self.cap = cv2.VideoCapture("/dev/video5")  #my usb setup
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open camera")
        self.fw = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.fh = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.cx, self.cy = self.fw // 2, self.fh // 2

        # Optional output video recording
        self.rec: Optional[cv2.VideoWriter] = None

    
    def _detect_boat(self, frame: np.ndarray) -> Optional[Tuple[float, float, float, float]]:
        """Return highest confidence boat or None.
        
        Args:
            frame (np.ndarray): The input frame.
            
        Returns:
            Optional[Tuple[float, float, float, float]]: The bounding box
            coordinates (x1, y1, x2, y2) of the detected boat or None if no
            boat is detected."""
        results = self.model(frame, conf=self.cfg.CONF_THRESH)[0]
        boats = [b for b in results.boxes if int(b.cls) == 0]
        if not boats:
            return None
        best = max(boats, key=lambda b: float(b.conf))
        return tuple(map(float, best.xyxy[0]))  # type: ignore[arg-type]

    
    def _draw_overlay(self, frame: np.ndarray, zoomed: np.ndarray, zoom: float) -> None:
        """ Overlays for the zoomed image and the original frame. 
        
        Args:
            frame (np.ndarray): The original frame.
            zoomed (np.ndarray): The zoomed image.
            zoom (float): The current zoom level.
            
        Returns:
            None
        
        """ 
        cv2.drawMarker(frame, (self.cx, self.cy), (0, 0, 255), cv2.MARKER_CROSS, 20, 1)
        # Zoom label drawn on the  image so it remains readable
        text = f"Zoom {zoom:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(zoomed, (10, 10), (10 + tw + 10, 10 + th + 10), (0, 0, 0), -1)
        cv2.putText(zoomed, text, (15, 10 + th), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    
    def run(self) -> None:
        """Main loop."""
        cfg = self.cfg

        # Optional MP4 recording 
        self.rec = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, (self.fw, self.fh))

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            bbox = self._detect_boat(frame)
            if bbox:
                x1, y1, x2, y2 = bbox
                bx, by = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                bw = x2 - x1

                # Smooth values over time
                self.smoother.push(bx, by, bw)
                sm_bx, sm_by, sm_bw = self.smoother.mean()

                # Zoom decision based on smoothed width
                self.zoom_ctl.update(sm_bw, self.fw)

                # Pid corrections
                err_x, err_y = sm_bx - self.cx, sm_by - self.cy
                d_pan = self.pid_pan.update(err_x) if abs(err_x) > cfg.DEAD_ZONE_PX else 0.0
                d_tilt = self.pid_tilt.update(err_y) if abs(err_y) > cfg.DEAD_ZONE_PX else 0.0

                # Clamp step size
                d_pan = float(np.clip(d_pan, -cfg.MAX_STEP_DEG, cfg.MAX_STEP_DEG))
                d_tilt = float(np.clip(d_tilt, -cfg.MAX_STEP_DEG, cfg.MAX_STEP_DEG))
                self.servo.update(d_pan, d_tilt)

                # Visualise bbox
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.circle(frame, (int(sm_bx), int(sm_by)), 4, (0, 255, 0), -1)
            else:
                self.zoom_ctl.relax()

            # Draw the centre marker and zoom label
            zoomed = ZoomController.apply(frame, self.zoom_ctl.level)
            self._draw_overlay(frame, zoomed, self.zoom_ctl.level)

            if self.rec is not None:
                self.rec.write(zoomed)

            cv2.imshow("PTZ Boat Tracker", zoomed)
            if cv2.waitKey(1) and 0xFF == ord("q"):
                break

        # Cleanup
        self.cap.release()
        self.servo.close()
        if self.rec is not None:
            self.rec.release()
        cv2.destroyAllWindows()



if __name__ == "__main__":
    PTZTracker().run()


