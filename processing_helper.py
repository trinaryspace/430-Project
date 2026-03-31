import numpy as np

def ca_cfar_1d(signal, guard_cells, train_cells, threshold_factor):
    """
    1D Cell-Averaging CFAR.
    Parameters:
      signal: 1D array of magnitude (or dB).
      guard_cells: Number of guard cells on each side.
      train_cells: Number of training cells on each side.
      threshold_factor: Detection threshold offset (added since signal is in dB).
    Returns:
      detections: list of indices where target is detected.
    """
    num_cells = len(signal)
    detections = []
    
    start_idx = train_cells + guard_cells
    end_idx = num_cells - (train_cells + guard_cells)
    
    if start_idx >= end_idx:
        return detections
        
    for i in range(start_idx, end_idx):
        left_train = signal[i - guard_cells - train_cells : i - guard_cells]
        right_train = signal[i + guard_cells + 1 : i + guard_cells + train_cells + 1]
        
        noise_level = np.mean(np.concatenate([left_train, right_train]))
        threshold = noise_level + threshold_factor
        
        if signal[i] > threshold:
            detections.append(i)
            
    # Group contiguous detections and pick the max
    grouped_detections = []
    if detections:
        current_group = [detections[0]]
        for j in range(1, len(detections)):
            if detections[j] == detections[j-1] + 1:
                current_group.append(detections[j])
            else:
                peak_idx = current_group[np.argmax([signal[idx] for idx in current_group])]
                grouped_detections.append(peak_idx)
                current_group = [detections[j]]
                
        peak_idx = current_group[np.argmax([signal[idx] for idx in current_group])]
        grouped_detections.append(peak_idx)
        
    return grouped_detections

class TargetTrack:
    def __init__(self, target_id, pos, dt, range_val=0.0, freq_val=0.0, angle=0.0, **kwargs):
        self.target_id = target_id
        self.pos = pos
        self.vel = 0.0
        self.dt = dt
        self.age = 1
        self.coasts = 0
        self.range_val = range_val
        self.freq_val = freq_val
        self.angle = angle
        self.mag = kwargs.get('mag', 0.0)
        self.is_tag = kwargs.get('is_tag', False)
        self.mod_freq = kwargs.get('mod_freq', 0.0)
        self.is_part_of_tag = False

class Tracker:
    def __init__(self, alpha=0.5, beta=0.1, max_coasts=5, max_dist=2.0, dt=0.5, min_hits=3):
        """
        Simple Alpha-Beta tracker for 1D range targets.
        """
        self.alpha = alpha
        self.beta = beta
        self.max_coasts = max_coasts
        self.max_dist = max_dist
        self.dt = dt
        self.min_hits = min_hits
        
        self.tracks = []
        self.next_id = 1
        
    def update(self, detections):
        """
        Update tracks with new detections.
        detections: list of dictionaries, e.g., [{'pos': pos, 'range': r, 'freq': f, 'angle': a}, ...]
        """
        # Predict
        for track in self.tracks:
            track.pos += track.vel * self.dt
            track.coasts += 1
            
        unmatched_detections = list(detections)
        
        # Match detections to tracks
        for track in self.tracks:
            if not unmatched_detections:
                break
            
            distances = [abs(d['pos'] - track.pos) for d in unmatched_detections]
            min_idx = np.argmin(distances)
            
            if distances[min_idx] < self.max_dist:
                match = unmatched_detections.pop(min_idx)
                
                # Update (Alpha-Beta filter)
                residual = match['pos'] - track.pos
                track.pos = track.pos + self.alpha * residual
                track.vel = track.vel + (self.beta / self.dt) * residual
                
                # Update auxiliary data
                track.range_val = match['range']
                track.freq_val = match['freq']
                if 'mag' in match:
                    track.mag = match['mag']
                
                # Just update to the most recent angle measurement
                track.angle = match['angle']
                track.age += 1
                track.coasts = 0
                
        # Create new tracks for unmatched detections
        for d in unmatched_detections:
            self.tracks.append(TargetTrack(
                self.next_id, 
                d['pos'], 
                self.dt,
                range_val=d['range'],
                freq_val=d['freq'],
                angle=d['angle'],
                mag=d.get('mag', 0.0)
            ))
            self.next_id += 1
            
        # Delete dead tracks
        self.tracks = [t for t in self.tracks if t.coasts <= self.max_coasts]
        
        # Return currently active tracks that have met the minimum hit count
        return [t for t in self.tracks if t.age >= self.min_hits]

def analyze_frame(fft_rx0, fft_rx1, fft_comp_rx0, fft_comp_rx1, freq_axis, x_for_cfar, params, tracker, active_tags, next_tag_id):
    """
    Core logic for frame analysis (CA-CFAR, AoA, Tracking, Tag Generation).
    """
    from itertools import combinations
    
    avg_fft = 20*np.log10((10**(fft_rx0/20) + 10**(fft_rx1/20)) / 2)
    avg_fft[avg_fft < -40] = -40
    det_bins = ca_cfar_1d(avg_fft, guard_cells=30, train_cells=128, threshold_factor=6.50)
    
    # Ignore frequencies between -150 kHz and 150 kHz (clutter)
    actual_freqs = np.fft.fftshift(freq_axis)
    det_bins = [i for i in det_bins if abs(actual_freqs[i]) > 150000.0]
    
    det_x = np.array([x_for_cfar[i] for i in det_bins])
    det_y = np.array([avg_fft[i] for i in det_bins])
    
    c = 299792458
    bw_khz = float(params.get('Upper_RF_Frequency_kHz', 0)) - float(params.get('Lower_RF_Frequency_kHz', 0))
    tc_sec = float(params.get('Chirp_Time_sec', 0))
    slope = (bw_khz * 1000) / tc_sec if tc_sec > 0 else 1
    
    detections = []
    print("number of detections: ", len(det_bins))
    for i, bin_idx in enumerate(det_bins):
        # Phase Difference Calculation
        rx0_c = fft_comp_rx0[bin_idx]
        rx1_c = fft_comp_rx1[bin_idx]
        phase_diff = np.angle(rx1_c * np.conj(rx0_c)) # correction factor needed 2.39?
        sin_theta = np.clip(phase_diff / np.pi, -1.0, 1.0)
        angle_deg = np.degrees(np.arcsin(sin_theta)) +63
        print("pre-tracked target angle: ", angle_deg)
        
        freq_val = abs(actual_freqs[bin_idx])
        range_val = (c * freq_val) / (2 * slope) if slope > 0 else 0
        
        detections.append({
            'pos': det_x[i],
            'range': range_val,
            'freq': freq_val,
            'angle': angle_deg,
            'mag': avg_fft[bin_idx]
        })
        
    tracks = tracker.update(detections)
    
    MAG_THRESHOLD = 3.0
    ANGLE_THRESHOLD = 10.0
    
    for t in tracks:
        t.is_part_of_tag = False
        
    # Sort tracks by range
    tracks.sort(key=lambda t: t.range_val)
    
    tags_this_frame = []
    current_frame_tag_keys = set()
    
    i = 0
    print(f"Number of tracks: {len(tracks)}")
    while i < len(tracks) - 1:
        t1 = tracks[i]
        t2 = tracks[i+1]
        
        print(f"Comparing targets {t1.target_id} and {t2.target_id}")
        print(f"Target {t1.target_id} mag: {t1.mag}, angle: {t1.angle}")
        print(f"Target {t2.target_id} mag: {t2.mag}, angle: {t2.angle}")
        
        db_diff = abs(t1.mag - t2.mag)
        angle_diff = abs(t1.angle - t2.angle)
        
        # Check if the pair meets DB matching and angle inversion criteria
        if db_diff <= MAG_THRESHOLD and angle_diff <= ANGLE_THRESHOLD:
            t1.is_part_of_tag = True
            t2.is_part_of_tag = True
            
            pair_key = tuple(sorted((t1.target_id, t2.target_id)))
            current_frame_tag_keys.add(pair_key)
            
            if pair_key not in active_tags:
                active_tags[pair_key] = next_tag_id
                next_tag_id += 1
                
            tag_id = active_tags[pair_key]
            
            tag_range = abs(t1.range_val - t2.range_val) / 2.0
            tag_freq_mid = (t1.range_val + t2.range_val) / 2.0 
            mod_freq = (2 * tag_freq_mid * slope) / c if slope > 0 else 0
            tag_angle = (t1.angle + t2.angle) / 2.0
            tag_pos = abs(t1.pos - t2.pos) / 2.0
            
            tag_obj = TargetTrack(
                target_id=tag_id,
                pos=tag_pos,
                dt=0.1,
                range_val=tag_range,
                freq_val=abs(t1.freq_val - t2.freq_val) / 2.0,
                angle=tag_angle,
                mag=(t1.mag + t2.mag) / 2.0,
                is_tag=True,
                mod_freq=mod_freq
            )
            tags_this_frame.append(tag_obj)
            i += 2
        else:
            i += 1
            
    active_tags_out = {k: v for k, v in active_tags.items() if k in current_frame_tag_keys}
    
    return det_x, det_y, tracks, tags_this_frame, active_tags_out, next_tag_id, avg_fft
