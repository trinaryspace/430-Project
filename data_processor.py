import time
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

def safe_read_data(filepath, retries=10, delay=0.05):
    """
    Safely reads radar data frames by implementing a retry loop.
    This prevents PermissionError or IOError if the file is still locked
    by the OS/GUI while it's being written.
    """
    for attempt in range(retries):
        try:
            # We open in binary read mode 'rb', adjust to 'r' if it's purely text data
            with open(filepath, 'rb') as f:
                data = f.read()
                return data
        except (PermissionError, IOError) as e:
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                print(f"Failed to read file {filepath} after {retries} attempts: {e}")
                return None

def process_radar_frame(data):
    """
    Simulates custom signal processing on the freshly read radar data frame.
    """
    print("Processing frame...")
    time.sleep(0.2)
    # TODO: Add your custom signal processing here

class RadarDataHandler(FileSystemEventHandler):
    """
    Watchdog event handler that processes new or modified files in the radar data directory.
    """
    def on_created(self, event):
        # Triggered when a file or directory is created
        if not event.is_directory:
            self._handle_event(event.src_path)

    def on_modified(self, event):
        # Triggered when a file or directory is modified
        # Note: Depending on the OS, a file creation can trigger both on_created and on_modified.
        if not event.is_directory:
            self._handle_event(event.src_path)

    def _handle_event(self, filepath):
        if not filepath.lower().endswith('.txt'):
            return
        print(f"Detected event on: {filepath}")
        data = safe_read_data(filepath)
        if data is not None:
            process_radar_frame(data)

def start_pipeline(monitor_dir="./recordings"):
    """
    Sets up the watchdog observer and keeps the script running to monitor the directory.
    """
    
    # Ensure the target directory exists
    if not os.path.exists(monitor_dir):
        os.makedirs(monitor_dir)
        print(f"Created directory: {monitor_dir}")

    event_handler = RadarDataHandler()
    observer = Observer()
    
    # Schedule the observer to monitor the specific directory
    observer.schedule(event_handler, monitor_dir, recursive=False)
    observer.start()
    
    print(f"Monitoring '{monitor_dir}' for new radar data files...")
    print("Press Ctrl+C to stop.")
    
    try:
        # Keep the main thread alive while the observer runs in the background
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        print("\nStopping data processing pipeline.")
    
    observer.join()

if __name__ == "__main__":
    start_pipeline()
