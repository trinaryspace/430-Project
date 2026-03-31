import sys
import os
import time
import argparse
import json
import datetime
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets, QtGui
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from datadecoder_txt import parse_radar_file, process_radar_data
from processing_helper import Tracker, analyze_frame

class SignalEmitter(QtCore.QObject):
    file_detected = QtCore.pyqtSignal(str)

class RadarDataHandler(FileSystemEventHandler):
    def __init__(self, emitter):
        super().__init__()
        self.emitter = emitter

    def on_created(self, event):
        if not event.is_directory:
            self._handle_event(event.src_path)

    def on_modified(self, event):
        if not event.is_directory:
            self._handle_event(event.src_path)

    def _handle_event(self, filepath):
        if not filepath.lower().endswith('.txt'):
            return
            
        # Retry loop to ensure file is fully written before emitting
        retries = 10
        delay = 0.05
        for attempt in range(retries):
            try:
                # Attempt to open file to ensure it's fully accessible
                with open(filepath, 'r') as f:
                    pass
                # success
                self.emitter.file_detected.emit(filepath)
                return
            except (PermissionError, IOError) as e:
                if attempt < retries - 1:
                    time.sleep(delay)
                else:
                    print(f"Failed to read file {filepath} after {retries} attempts: {e}")

class RealtimeRadarApp(QtWidgets.QMainWindow):
    def __init__(self, plot_hz=False):
        super().__init__()
        self.plot_hz = plot_hz
        
        # Tracking State persists across updates
        max_assoc_dist = 500.0 if plot_hz else 2.0
        self.tracker = Tracker(alpha=0.8, beta=0.01, max_coasts=3, max_dist=max_assoc_dist, dt=0.5, min_hits=3)
        self.next_tag_id = 1000
        self.active_tags = {}
        self.frame_counter = 0
        
        # Create a session-specific JSON log directory
        session_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.json_log_dir = os.path.join("json logs", f"session_{session_timestamp}")
        os.makedirs(self.json_log_dir, exist_ok=True)
        print(f"Logging JSON data to: {self.json_log_dir}")
        
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('Realtime Radar Tracker - Waiting for Data...')
        self.cw = pg.GraphicsLayoutWidget()
        self.setCentralWidget(self.cw)
        self.resize(1000, 700)
        
        self.setup_polar_plot()
        
        # Setup Raw Data Plot (Time Domain)
        self.raw_plot = self.cw.addPlot(title="Raw Data - Rx0")
        self.raw_plot.addLegend()
        self.raw_curve_real = self.raw_plot.plot(pen='y', name="Real")
        self.raw_curve_imag = self.raw_plot.plot(pen='c', name="Imag")
        self.raw_plot.setLabel('bottom', 'Sample Index')
        self.raw_plot.setLabel('left', 'Amplitude')
        
        self.cw.nextRow()
        
        # Setup Range FFT Plot (Frequency Domain)
        self.fft_plot = self.cw.addPlot(title="Range FFT - Rx0 & Rx1")
        self.fft_plot.addLegend()
        self.fft_curve_rx0 = self.fft_plot.plot(pen='g', name="Rx0")
        self.fft_curve_rx1 = self.fft_plot.plot(pen='m', name="Rx1")
        
        if self.plot_hz:
            self.fft_plot.setLabel('bottom', 'Frequency', units='Hz')
        else:
            self.fft_plot.setLabel('bottom', 'Range', units='m')
            
        self.fft_plot.setLabel('left', 'Magnitude', units='dB')

        # Add Scatter plots for CFAR Detections and Tracks
        self.det_scatter = pg.ScatterPlotItem(size=8, pen=pg.mkPen(None), brush=pg.mkBrush('r'), name="Detections")
        self.trk_scatter = pg.ScatterPlotItem(size=12, pen=pg.mkPen('b', width=2), brush=None, symbol='+', name="Tracks")
        self.fft_plot.addItem(self.det_scatter)
        self.fft_plot.addItem(self.trk_scatter)

    def setup_polar_plot(self):
        self.polar_win = pg.GraphicsLayoutWidget(title="Polar Plot")
        self.polar_win.resize(600, 500)
        self.polar_win.setWindowTitle("Tracked Tags - Polar Map")
        self.polar_plot = self.polar_win.addPlot(title="Range-Angle Map (10m)")
        self.polar_plot.hideAxis('bottom')
        self.polar_plot.hideAxis('left')
        self.polar_plot.setRange(xRange=[-11, 11], yRange=[0, 11], padding=0.0)
        self.polar_plot.setAspectLocked(True)

        for r in range(2, 12, 2):
            theta = np.linspace(-np.pi/2, np.pi/2, 100)
            x = -r * np.sin(theta)
            y = r * np.cos(theta)
            self.polar_plot.plot(x, y, pen=pg.mkPen('gray', width=1, style=QtCore.Qt.DashLine))
            text = pg.TextItem(f"{r}", color='gray', anchor=(0.5, 1))
            self.polar_plot.addItem(text)
            text.setPos(r, 0)
            
        angles = [-90, -60, -30, 0, 30, 60, 90]
        for a in angles:
            rad_a = np.radians(a)
            x_line = [0, -10 * np.sin(rad_a)]
            y_line = [0, 10 * np.cos(rad_a)]
            self.polar_plot.plot(x_line, y_line, pen=pg.mkPen('gray', width=1, style=QtCore.Qt.DashLine))
            text = pg.TextItem(f"{a}°", color='gray', anchor=(0.5, 0.5))
            self.polar_plot.addItem(text)
            text.setPos(-10.5 * np.sin(rad_a), 10.5 * np.cos(rad_a))

        range_label = pg.TextItem("Range [m]", color='gray', anchor=(0.5, 0))
        self.polar_plot.addItem(range_label)
        range_label.setPos(8, -1.0)

        angle_label = pg.TextItem("Angle [°]", color='gray', anchor=(0.5, 1))
        self.polar_plot.addItem(angle_label)
        angle_label.setPos(0, 11)

        self.tag_polar_scatter = pg.ScatterPlotItem(size=15, pen=pg.mkPen('black'), brush=pg.mkBrush('b'), name="Tags")
        self.polar_plot.addItem(self.tag_polar_scatter)
        self.polar_win.show()

    def closeEvent(self, event):
        if hasattr(self, 'polar_win'):
            self.polar_win.close()
        super().closeEvent(event)

    def export_tags_per_frame_to_json(self, tags, frame_idx):
        """Exports tracked tags to a JSON file stamped with the current time for this frame."""
        current_time = datetime.datetime.now()
        timestamp_str = current_time.strftime("%Y%m%d_%H%M%S_%f")[:-3]  # up to milliseconds
        
        data = {
            "frame": frame_idx,
            "timestamp": timestamp_str,
            "num_tags": len(tags),
            "tags": [
                {
                    "id": tag.target_id,
                    "range_m": round(tag.range_val, 4),
                    "angle_deg": round(tag.angle, 4),
                    "mod_freq_hz": round(tag.mod_freq, 4),
                    "magnitude_db": round(tag.mag, 4)
                } for tag in tags
            ]
        }
        
        # Write out to a new JSON file for the frame
        filename = f"frame_{frame_idx:04d}_{timestamp_str}.json"
        filepath = os.path.join(self.json_log_dir, filename)
        
        try:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            print(f"Error writing JSON for frame {frame_idx}: {e}")

    @QtCore.pyqtSlot(str)
    def process_new_file(self, filepath):
        try:
            params, frames = parse_radar_file(filepath)
            if not frames:
                return
                
            processed_frames, range_ffts, range_ffts_complex, range_axis, freq_axis = process_radar_data(params, frames, n_fft=2**14)
            
            if not processed_frames:
                return
                
            self.setWindowTitle(f'Realtime Radar Tracker - Last File: {os.path.basename(filepath)}')
            
            # Using the first frame in the new file
            frame_idx = 0 
            
            raw_rx0 = processed_frames[frame_idx][0, 0, :]
            fft_rx0 = range_ffts[frame_idx][0, 0, :]
            fft_rx1 = range_ffts[frame_idx][1, 0, :]
            fft_comp_rx0 = range_ffts_complex[frame_idx][0, 0, :]
            fft_comp_rx1 = range_ffts_complex[frame_idx][1, 0, :]
            
            self.raw_curve_real.setData(np.real(raw_rx0))
            self.raw_curve_imag.setData(np.imag(raw_rx0))
            
            x_axis = freq_axis if self.plot_hz else range_axis
            
            if self.plot_hz:
                x_shifted = np.fft.fftshift(x_axis)
                self.fft_curve_rx0.setData(x_shifted, fft_rx0)
                self.fft_curve_rx1.setData(x_shifted, fft_rx1)
                x_for_cfar = x_shifted
            else:
                self.fft_curve_rx0.setData(x_axis, fft_rx0)
                self.fft_curve_rx1.setData(x_axis, fft_rx1)
                x_for_cfar = x_axis
                
            # Run shared processing functionality
            det_x, det_y, tracks, tags_this_frame, self.active_tags, self.next_tag_id, avg_fft = analyze_frame(
                fft_rx0, fft_rx1, fft_comp_rx0, fft_comp_rx1, 
                freq_axis, x_for_cfar, params, 
                self.tracker, self.active_tags, self.next_tag_id
            )
            
            if len(det_x) > 0:
                self.det_scatter.setData(det_x, det_y)
            else:
                self.det_scatter.setData([], [])
                
            trk_x = [t.pos for t in tracks]
            max_y = np.max(avg_fft) if len(avg_fft) > 0 else 0
            trk_y = [max_y] * len(trk_x)
            
            if len(trk_x) > 0:
                self.trk_scatter.setData(trk_x, trk_y)
                
                print(f"\n--- Realtime Frame {self.frame_counter:04d} (File: {os.path.basename(filepath)}) ---")
                for t in tracks:
                    tag_str = " (Tag Part)" if getattr(t, 'is_part_of_tag', False) else ""
                    print(f" Target ID {t.target_id:2d}: Range = {t.range_val:5.2f}m, "
                          f"Freq = {t.freq_val:8.1f}Hz, Angle = {t.angle:6.2f}deg, Mag = {t.mag:5.1f}dB{tag_str}")
                          
                for tag in tags_this_frame:
                    print(f" [TAG] ID {tag.target_id:2d}: Range = {tag.range_val:5.2f}m, "
                          f"Mod Freq = {tag.mod_freq:8.1f}Hz, Angle = {tag.angle:6.2f}deg, Mag = {tag.mag:5.1f}dB")
                          
                polar_pts = []
                for tag in tags_this_frame:
                    r = tag.range_val if tag.range_val <= 10.0 else 10.0
                    rad_a = np.radians(tag.angle)
                    x = -r * np.sin(rad_a)
                    y = r * np.cos(rad_a)
                    polar_pts.append({'pos': (x, y)})
                
                if len(polar_pts) > 0:
                    self.tag_polar_scatter.setData(polar_pts)
                else:
                    self.tag_polar_scatter.setData([])
            else:
                self.trk_scatter.setData([], [])
                self.tag_polar_scatter.setData([])
            
            # Export the tags for this frame to JSON
            self.export_tags_per_frame_to_json(tags_this_frame, self.frame_counter)
            
            self.frame_counter += 1
            
        except Exception as e:
            print(f"Error processing {filepath}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Realtime display of radar data from directory.")
    parser.add_argument("--monitor-dir", type=str, default="./recordings", help="Directory to monitor")
    parser.add_argument("--plot-hz", action="store_true", help="Plot FFT with Hz instead of meters")
    args = parser.parse_args()

    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
        
    window = RealtimeRadarApp(plot_hz=args.plot_hz)
    window.show()
    
    emitter = SignalEmitter()
    emitter.file_detected.connect(window.process_new_file)
    
    if not os.path.exists(args.monitor_dir):
        os.makedirs(args.monitor_dir)
        print(f"Created directory: {args.monitor_dir}")

    event_handler = RadarDataHandler(emitter)
    observer = Observer()
    observer.schedule(event_handler, args.monitor_dir, recursive=False)
    observer.start()
    
    print(f"Monitoring '{args.monitor_dir}' for new radar data files...")
    print("Close the PyQt window to exit.")
    
    try:
        sys.exit(app.exec())
    finally:
        observer.stop()
        observer.join()

if __name__ == '__main__':
    main()
