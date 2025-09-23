import os
import subprocess
import time
import signal
import sys

RECORD_NAME = 'experiment'
rec_proc = None
exp_proc = None

def stop_recorder(proc, timeout=10):
    """Try to stop ffmpeg cleanly, fallback to kill if needed."""
    if proc and proc.poll() is None:  # still running
        try:
            proc.send_signal(signal.SIGINT)   # graceful stop
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            print("⚠️ Recorder still alive, forcing kill...")
            proc.kill()
            proc.wait()

try:
    output_folder = os.path.join(os.path.expanduser("~"), "Videos", "IsaacLab")
    os.makedirs(output_folder, exist_ok=True)

    for exp_num in range(50):
        rec_proc = None
        exp_proc = None
        try:
            print(f"\n--- Starting experiment {exp_num} ---")

            # Start FFmpeg recording
            record_cmd = [
                "/usr/bin/ffmpeg",
                "-video_size", "3440x1440",
                "-framerate", "60",
                "-f", "x11grab",
                "-i", ":1",
                f"{output_folder}/{RECORD_NAME}_{exp_num}.mp4"
            ]
            print("Recording command:", " ".join(record_cmd))
            rec_proc = subprocess.Popen(record_cmd)
            time.sleep(1)  # Give FFmpeg time to initialize

            # Start the experiment
            exp_cmd = [
                "python", 
                "isaaclab_experiments/planning.py", 
                "--exp_num", str(exp_num)
            ]
            print("Experiment command:", " ".join(exp_cmd))
            exp_proc = subprocess.Popen(exp_cmd)

            # Wait for the experiment to finish
            exp_proc.wait()

        except Exception as e:
            print(f"⚠️ Experiment {exp_num} crashed: {e}")

        finally:
            # Always stop recorder, even if experiment crashed
            print("Stopping recording...")
            stop_recorder(rec_proc)
            print(f"✔️ Experiment {exp_num} finished and video saved.")

except KeyboardInterrupt:
    print("\nCaught Ctrl+C! Exiting...")
    if exp_proc:
        exp_proc.terminate()
    if rec_proc:
        stop_recorder(rec_proc)
    sys.exit(1)
