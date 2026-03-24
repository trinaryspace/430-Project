import sys
import argparse
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from processing_helper import ca_cfar_1d, Tracker

# ==========================================
# SECTION 1: PARSING
# ==========================================
def parse_radar_file(filepath):
    """
    Reads the file to extract metadata parameters and individual frames of raw floats.
    """
    params = {}
    frames = []
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    data_floats = []
    in_data_section = False
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Skip the generic header identifier
        if line.startswith('IFRT'):
            continue
            
        # Parse parameter keys or frame dividers
        if line.startswith('#'):
            if '=' in line and not line.startswith('###') and not line.startswith('##'):
                parts = line.split('=')
                if len(parts) == 2:
                    key = parts[0].strip('# ').strip()
                    val = parts[1].strip()
                    try:
                        params[key] = float(val) if '.' in val else int(val)
                    except ValueError:
                        pass
            
            # Every frame starts with a frame number
            if line.startswith('# Frame_Number'):
                in_data_section = True
                if data_floats:
                    frames.append(data_floats)
                    data_floats = []
            continue
            
        # Parse data floats if inside the block
        if in_data_section:
            try:
                data_floats.append(float(line))
            except ValueError:
                pass
                
    # Append the final frame
    if data_floats:
        frames.append(data_floats)
        
    return params, frames


# ==========================================
# SECTION 2: RAW DATA PROCESSING
# ==========================================
def process_radar_data(params, raw_frames_list, n_fft=2**13):
    """
    Converts sequential floats into complex numbers, reshapes them into 
    (Rx, Chirp, Sample) dimensions, and calculates the Range FFT.
    """
    num_rx = int(params.get('Num_Rx_Antennas', 2))
    num_chirps = int(params.get('Chirps_per_Frame', 1))
    num_samples = int(params.get('Samples_per_Chirp', 1))
    
    # Range axis calculation
    c = 299792458
    try:
        f_lower = float(params.get('Lower_RF_Frequency_kHz', 0)) * 1000
        f_upper = float(params.get('Upper_RF_Frequency_kHz', 0)) * 1000
        bw = f_upper - f_lower if f_upper > f_lower else 1
    except:
        bw = 1
        
    range_res = c / (2 * bw) if bw != 0 else 1
    # bin spacing for zero-padded FFT: R_res * (N_samples / n_fft)
    range_bin_spacing = range_res * (num_samples / n_fft)
    range_axis = np.arange(n_fft) * range_bin_spacing
    
    # Frequency axis calculation (Hz)
    fs = float(params.get('Sampling_Frequency_kHz', 0)) * 1000
    freq_axis = np.fft.fftfreq(n_fft, d=1/fs)
    
    processed_frames = []
    range_ffts = []
    range_ffts_complex = []
    
    # Hanning window to reduce FFT leakage
    window = np.hanning(num_samples)
    
    for frame in raw_frames_list:
        arr = np.array(frame)
        
        try:
            # Reshape into (Chirp, Rx, Real/Imag, Samples)
            # Layout: Rx0 Real, Rx0 Imag, Rx1 Real, Rx1 Imag...
            arr = arr.reshape((num_chirps, num_rx, 2, num_samples))
            
            # Combine real and imaginary parts
            # arr[:, :, 0, :] is Real, arr[:, :, 1, :] is Imag
            complex_arr = arr[:, :, 0, :] + 1j * arr[:, :, 1, :]
            
            # Transpose to (Num_Rx, Num_Chirps, Num_Samples)
            arr = complex_arr.transpose(1, 0, 2)
            
        except ValueError as e:
            print(f"Warning: Shape mismatch in frame: {e}, skipping.")
            continue
            
        processed_frames.append(arr)
        
        # Calculate Range FFT (along the time dimension, axis=-1)
        # Apply window to each chirp
        windowed_data = arr * window
        fft_data = np.fft.fftshift(np.fft.fft(windowed_data, n=n_fft, axis=-1))
        
        # Convert to dB scaling: 20*log10(abs(fft))
        fft_mag_db = 20 * np.log10(np.abs(fft_data) + 1e-9)
        
        range_ffts.append(fft_mag_db)
        range_ffts_complex.append(fft_data)
        
    return processed_frames, range_ffts, range_ffts_complex, range_axis, freq_axis


# ==========================================
# SECTION 3: PLOTTING / PLAYBACK
# ==========================================
class RadarPlayer(QtWidgets.QMainWindow):
    def __init__(self, raw_frames, fft_frames, fft_complex_frames, range_axis, freq_axis, params, plot_hz=False):
        super().__init__()
        self.raw_frames = raw_frames
        self.fft_frames = fft_frames
        self.fft_complex_frames = fft_complex_frames
        self.range_axis = range_axis
        self.freq_axis = freq_axis
        self.params = params
        self.plot_hz = plot_hz
        self.num_frames = len(raw_frames)
        self.current_frame = 0
        self.is_paused = False
        
        # Initialize Tracker. Max distance depends on axis units.
        max_assoc_dist = 500.0 if plot_hz else 2.0
        self.tracker = Tracker(alpha=0.3, beta=0.01, max_coasts=3, max_dist=max_assoc_dist, dt=0.1, min_hits = 5)
        self.next_tag_id = 1000
        self.active_tags = {}
        
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('Radar Data Playback')
        self.cw = pg.GraphicsLayoutWidget()
        self.setCentralWidget(self.cw)
        self.resize(1000, 700)
        
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
        
        # Start Timer for Playback Loop
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plots)
        self.timer.start(100) # Update every 100 ms (10 FPS)
        
    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Space:
            self.is_paused = not self.is_paused
            status = "Paused" if self.is_paused else "Playing"
            self.setWindowTitle(f'Radar Data Playback - {status}')
        super().keyPressEvent(event)
        
    def update_plots(self):
        if self.is_paused:
            return
            
        if self.current_frame >= self.num_frames:
            self.current_frame = 0 # Loop back to the beginning
            
        # Extract the first chirp for both RX antennas (Rx, Chirp, Sample)
        raw_rx0 = self.raw_frames[self.current_frame][0, 0, :]
        fft_rx0 = self.fft_frames[self.current_frame][0, 0, :]
        fft_rx1 = self.fft_frames[self.current_frame][1, 0, :]
        fft_comp_rx0 = self.fft_complex_frames[self.current_frame][0, 0, :]
        fft_comp_rx1 = self.fft_complex_frames[self.current_frame][1, 0, :]
        
        # Update plotted curves
        self.raw_curve_real.setData(np.real(raw_rx0))
        self.raw_curve_imag.setData(np.imag(raw_rx0))
        
        x_axis = self.freq_axis if self.plot_hz else self.range_axis
        
        # For FFT plots, typically we only show the positive frequencies if it's real data, 
        # but here we have complex data. However, Range FFT is usually shown for positive range.
        # If plot_hz is True, we might want to show the full spectrum or just positive.
        # Given the context of "Range FFT", usually we look at positive values.
        
        if self.plot_hz:
            # Show full frequency spectrum
            x_shifted = np.fft.fftshift(x_axis)
            self.fft_curve_rx0.setData(x_shifted, fft_rx0)
            self.fft_curve_rx1.setData(x_shifted, fft_rx1)
            x_for_cfar = x_shifted
        else:
            self.fft_curve_rx0.setData(x_axis, fft_rx0)
            self.fft_curve_rx1.setData(x_axis, fft_rx1)
            x_for_cfar = x_axis
            
        # CFAR and Tracking on averaged Rx0 and Rx1
        avg_fft = (fft_rx0 + fft_rx1) / 2.0
        # Use ~40 bin peak width -> 30 guard, 40 train, 15dB threshold
        det_bins = ca_cfar_1d(avg_fft, guard_cells=40, train_cells=80, threshold_factor=6.50)
        
        # Ignore frequencies between -150 kHz and 150 kHz (clutter)
        actual_freqs = np.fft.fftshift(self.freq_axis)
        det_bins = [i for i in det_bins if abs(actual_freqs[i]) > 150000.0]
        
        det_x = np.array([x_for_cfar[i] for i in det_bins])
        det_y = np.array([avg_fft[i] for i in det_bins])
        
        if len(det_x) > 0:
            self.det_scatter.setData(det_x, det_y)
        else:
            self.det_scatter.setData([], [])
            
        c = 299792458
        fs = float(self.params.get('Sampling_Frequency_kHz', 0)) * 1000
        bw_khz = float(self.params.get('Upper_RF_Frequency_kHz', 0)) - float(self.params.get('Lower_RF_Frequency_kHz', 0))
        tc_sec = float(self.params.get('Chirp_Time_sec', 0))
        slope = (bw_khz * 1000) / tc_sec if tc_sec > 0 else 1
            
        detections = []
        for i, bin_idx in enumerate(det_bins):
            # Phase Difference Calculation
            rx0_c = fft_comp_rx0[bin_idx]
            rx1_c = fft_comp_rx1[bin_idx]
            phase_diff = np.angle(rx1_c * np.conj(rx0_c)) + 2.39 #needs some correction factor?
            sin_theta = np.clip(phase_diff / np.pi, -1.0, 1.0)
            angle_deg = np.degrees(np.arcsin(sin_theta))
            
            freq_val = abs(actual_freqs[bin_idx])
            range_val = (c * freq_val) / (2 * slope)
            
            detections.append({
                'pos': det_x[i],
                'range': range_val,
                'freq': freq_val,
                'angle': angle_deg,
                'mag': avg_fft[bin_idx]
            })
            
        tracks = self.tracker.update(detections)
        trk_x = [t.pos for t in tracks]
        max_y = np.max(avg_fft) if len(avg_fft) > 0 else 0
        trk_y = [max_y] * len(trk_x)
        
        if len(trk_x) > 0:
            self.trk_scatter.setData(trk_x, trk_y)
            
            # --- Tag Detection Logic ---
            from itertools import combinations
            from processing_helper import TargetTrack
            
            MAG_THRESHOLD = 3.0
            ANGLE_THRESHOLD = 5.0
            
            for t in tracks:
                t.is_part_of_tag = False
                
            tags_this_frame = []
            current_frame_tag_keys = set()
            
            for t1, t2 in combinations(tracks, 2):
                if abs(t1.mag - t2.mag) <= MAG_THRESHOLD and abs(t1.angle - t2.angle) <= ANGLE_THRESHOLD:
                    t1.is_part_of_tag = True
                    t2.is_part_of_tag = True
                    
                    pair_key = tuple(sorted((t1.target_id, t2.target_id)))
                    current_frame_tag_keys.add(pair_key)
                    
                    if pair_key not in self.active_tags:
                        self.active_tags[pair_key] = self.next_tag_id
                        self.next_tag_id += 1
                        
                    tag_id = self.active_tags[pair_key]
                    mod_freq = (t1.freq_val + t2.freq_val) / 2.0
                    fb = 1.0 * abs(t1.freq_val - t2.freq_val)
                    tag_range = (c * fb) / (2 * slope) if slope > 0 else 0
                    tag_angle = (t1.angle + t2.angle) / 2.0
                    tag_pos = (t1.pos + t2.pos) / 2.0
                    
                    tag_obj = TargetTrack(
                        target_id=tag_id,
                        pos=tag_pos,
                        dt=0.1,
                        range_val=tag_range,
                        freq_val=fb,
                        angle=tag_angle,
                        mag=(t1.mag + t2.mag) / 2.0,
                        is_tag=True,
                        mod_freq=mod_freq
                    )
                    tags_this_frame.append(tag_obj)
                    
            self.active_tags = {k: v for k, v in self.active_tags.items() if k in current_frame_tag_keys}
            
            print(f"--- Frame {self.current_frame:04d} ---")
            for t in tracks:
                tag_str = " (Tag Part)" if getattr(t, 'is_part_of_tag', False) else ""
                print(f" Target ID {t.target_id:2d}: Range = {t.range_val:5.2f}m, "
                      f"Freq = {t.freq_val:8.1f}Hz, Angle = {t.angle:6.2f}deg, Mag = {t.mag:5.1f}dB{tag_str}")
                      
            for tag in tags_this_frame:
                print(f" [TAG] ID {tag.target_id:2d}: Range = {tag.range_val:5.2f}m, "
                      f"Mod Freq = {tag.mod_freq:8.1f}Hz, Angle = {tag.angle:6.2f}deg, Mag = {tag.mag:5.1f}dB")
        else:
            self.trk_scatter.setData([], [])
        
        self.current_frame += 1

def play_radar(filepath, plot_hz=False):
    print(f"Parsing file: {filepath} ...")
    params, frames = parse_radar_file(filepath)
    
    if not frames:
        print("Error: No frames found in the specified file.")
        sys.exit(1)
        
    # Print radar waveform parameters
    bw_khz = float(params.get('Upper_RF_Frequency_kHz', 0)) - float(params.get('Lower_RF_Frequency_kHz', 0))
    bw_mhz = bw_khz / 1000.0
    tc_sec = float(params.get('Chirp_Time_sec', 0))
    tc_usec = tc_sec * 1e6
    fs_khz = float(params.get('Sampling_Frequency_kHz', 0))
    n_samples = int(params.get('Samples_per_Chirp', 0))
    n_chirps = int(params.get('Chirps_per_Frame', 0))
    
    slope = bw_mhz / tc_usec if tc_usec > 0 else 0
    c = 299792458
    r_max = (c * n_samples) / (4 * bw_mhz * 1e6) if bw_mhz > 0 else 0
    
    print("\n" + "="*40)
    print("RADAR WAVEFORM INFORMATION")
    print("="*40)
    print(f"Bandwidth:       {bw_mhz:.2f} MHz")
    print(f"Chirp Duration:  {tc_usec:.2f} us")
    print(f"Chirp Slope:     {slope:.2f} MHz/us")
    print(f"Sampling Rate:   {fs_khz:.2f} kHz")
    print(f"Max Range:       {r_max:.2f} m")
    print(f"Samples/Chirp:   {n_samples}")
    print(f"Chirps/Frame:    {n_chirps}")
    print("="*40 + "\n")

    print("Processing radar frames...")
    processed_frames, range_ffts, range_ffts_complex, range_axis, freq_axis = process_radar_data(params, frames, n_fft=2**14)
    
    print(f"Total Frames Processed: {len(processed_frames)}")
    print("Launching Playback Window...")
    
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
        
    player = RadarPlayer(processed_frames, range_ffts, range_ffts_complex, range_axis, freq_axis, params, plot_hz=plot_hz)
    player.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    # Set up argparse for command line execution
    parser = argparse.ArgumentParser(description="Parse and playback FMCW radar raw data.")
    parser.add_argument(
        "filepath", 
        type=str, 
        help="Path to the radar raw data file (e.g., Position2Go_record.raw.txt)"
    )
    
    parser.add_argument(
        "--plot-hz",
        action="store_true",
        help="Plot the Range FFT with Hz instead of meters"
    )
    
    args = parser.parse_args()
    
    play_radar(args.filepath, plot_hz=args.plot_hz)