"""Command construction and process management for a single experiment run."""

import sys

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, Signal

from app.config import ENV_VAR
from app.registry import PLANNING_SCRIPT, PROJECT_ROOT


def build_command(run):
    """Translate the launcher selection into ``planning.py`` arguments.

    ``run`` keys: task, space, exp_num, num_envs, headless, video, video_length,
    log, follow_camera.
    """
    args = [
        str(PLANNING_SCRIPT.relative_to(PROJECT_ROOT)),
        "--task", run["task"],
        "--space", run["space"],
        "--exp_num", str(run["exp_num"]),
        "--num_envs", str(run["num_envs"]),
        "--log", str(run["log"]),
        "--follow_camera", str(run["follow_camera"]),
    ]
    if run["headless"]:
        args.append("--headless")
    if run["video"]:
        args += ["--video", "--video_length", str(run["video_length"])]
    return args


class SimulationRunner(QObject):
    """Runs ``planning.py`` in a child process and relays its output."""

    output = Signal(str)
    started = Signal()
    finished = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.process = QProcess(self)
        self.process.setWorkingDirectory(str(PROJECT_ROOT))
        self.process.setProcessChannelMode(QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._relay)
        self.process.started.connect(self.started)
        self.process.finished.connect(lambda code, _status: self.finished.emit(code))

    def is_running(self):
        return self.process.state() != QProcess.NotRunning

    def start(self, args, config_path):
        env = QProcessEnvironment.systemEnvironment()
        env.insert(ENV_VAR, str(config_path))
        # stdout is a pipe here, so Python would block-buffer it; force line-by-line output
        env.insert("PYTHONUNBUFFERED", "1")
        self.process.setProcessEnvironment(env)
        self.process.start(sys.executable, ["-u"] + args)

    def stop(self):
        if not self.is_running():
            return
        self.process.terminate()
        if not self.process.waitForFinished(5000):
            self.process.kill()

    def _relay(self):
        data = self.process.readAllStandardOutput().data().decode("utf-8", errors="replace")
        if data:
            self.output.emit(data)
