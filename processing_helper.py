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
    def __init__(self, alpha=0.5, beta=0.1, alpha_angle=0.2, max_coasts=5, max_dist=2.0, dt=0.1, min_hits=3):
        """
        Simple Alpha-Beta tracker for 1D range targets.
        """
        self.alpha = alpha
        self.beta = beta
        self.alpha_angle = alpha_angle
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
                
                # Simple alpha filter for angle smoothing to reduce erratic measurements
                angle_residual = match['angle'] - track.angle
                track.angle = track.angle + self.alpha_angle * angle_residual
                
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
