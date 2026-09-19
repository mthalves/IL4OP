"""IL4OP experiment launcher: configure and run a single planning experiment."""

import shlex
import sys
from pathlib import Path

from PySide6.QtCore import QLocale, Qt
from PySide6.QtGui import QColor, QKeySequence, QPalette, QShortcut, QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLayout,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from app.config import write_agent_config
from app.registry import Planners, Problems
from app.runner import SimulationRunner, build_command
from app.widgets import ParameterForm, Section, StatusPill, expanding, field_label

APP_DIR = Path(__file__).resolve().parent
THEME = APP_DIR / "theme.qss"
ICONS = APP_DIR / "icons"

def link_button(text):
    button = QPushButton(text)
    button.setObjectName("link")
    button.setCursor(Qt.PointingHandCursor)
    return button


class MainWindow(QMainWindow):

    def __init__(self, problems, planners):
        super().__init__()
        self.problems = problems
        self.planners = planners
        self.runner = SimulationRunner(self)

        self.setWindowTitle("IL4OP Experiment Launcher")
        self.resize(1120, 800)
        self._build_ui()
        self._connect()
        self._on_scenario_changed()
        self._set_status("Ready", "idle")

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(20, 20, 20, 20)
        root.setSpacing(16)

        # Left column: configuration
        panel = QWidget()
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(0, 0, 4, 0)
        panel_layout.setSpacing(12)
        panel_layout.setSizeConstraint(QLayout.SetMinimumSize)
        panel_layout.addWidget(self._experiment_section())
        panel_layout.addWidget(self._parameters_section())
        panel_layout.addWidget(self._run_options_section())
        panel_layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFixedWidth(400)
        root.addWidget(scroll)

        # Right column: command, output, actions
        right = QVBoxLayout()
        right.setSpacing(12)

        self.copy_button = link_button("Copy")
        command_section = Section("Command", action=self.copy_button)
        self.command = QLineEdit()
        self.command.setObjectName("command")
        self.command.setReadOnly(True)
        command_section.body.addWidget(self.command)
        right.addWidget(command_section)

        self.clear_button = link_button("Clear")
        output_section = Section("Simulator output", action=self.clear_button)
        self.console = QPlainTextEdit()
        self.console.setObjectName("console")
        self.console.setReadOnly(True)
        self.console.setMaximumBlockCount(5000)
        self.console.setLineWrapMode(QPlainTextEdit.NoWrap)
        output_section.body.addWidget(self.console)
        right.addWidget(output_section, stretch=1)

        actions = QHBoxLayout()
        actions.setSpacing(10)
        self.status = StatusPill()
        self.stop_button = QPushButton("Stop")
        self.stop_button.setObjectName("danger")
        self.stop_button.setEnabled(False)
        self.start_button = QPushButton("Start experiment")
        self.start_button.setObjectName("primary")
        self.start_button.setCursor(Qt.PointingHandCursor)
        self.start_button.setToolTip("Ctrl+Return")
        actions.addWidget(self.status)
        actions.addStretch()
        actions.addWidget(self.stop_button)
        actions.addWidget(self.start_button)
        right.addLayout(actions)

        root.addLayout(right, stretch=1)

    def _experiment_section(self):
        section = Section("Experiment")
        form = section.add_form()

        self.scenario_combo = expanding(QComboBox())
        self.scenario_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        for problem_id, problem in self.problems.items.items():
            for scenario_id, scenario in problem["scenarios"].items():
                self.scenario_combo.addItem(
                    f"{problem['name']}  /  {scenario['name']}",
                    userData=(problem_id, scenario_id),
                )
        for index in range(self.scenario_combo.count()):
            if self.scenario_combo.itemData(index)[1] == "discrete":
                self.scenario_combo.setCurrentIndex(index)
                break
        form.addRow(field_label("Problem"), self.scenario_combo)

        self.planner_combo = expanding(QComboBox())
        self.planner_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        form.addRow(field_label("Planner"), self.planner_combo)
        return section

    def _parameters_section(self):
        self.reset_button = link_button("Reset")
        section = Section("Planner parameters", action=self.reset_button)
        self.parameters = ParameterForm()
        section.body.addWidget(self.parameters)
        return section

    def _run_options_section(self):
        section = Section("Run options")
        form = section.add_form()

        self.exp_num = expanding(QSpinBox())
        self.exp_num.setRange(0, 100_000)
        self.exp_num.setToolTip("Identifier used in the log file name")
        form.addRow(field_label("Experiment ID"), self.exp_num)

        self.num_envs = expanding(QSpinBox())
        self.num_envs.setRange(1, 1024)
        self.num_envs.setValue(1)
        form.addRow(field_label("Environments"), self.num_envs)

        self.headless = QCheckBox("Run headless (no viewport)")
        self.log = QCheckBox("Log results to logs/")
        self.follow_camera = QCheckBox("Camera follows the robot")
        self.follow_camera.setChecked(True)
        self.video = QCheckBox("Record video")
        for box in (self.headless, self.log, self.follow_camera, self.video):
            section.body.addWidget(box)

        video_form = section.add_form()
        self.video_length = expanding(QSpinBox())
        self.video_length.setRange(1, 100_000)
        self.video_length.setValue(200)
        self.video_length.setSuffix(" steps")
        self.video_length.setEnabled(False)
        video_form.addRow(field_label("Video length"), self.video_length)
        return section

    # ------------------------------------------------------------------
    # Behaviour
    # ------------------------------------------------------------------

    def _connect(self):
        self.scenario_combo.currentIndexChanged.connect(self._on_scenario_changed)
        self.planner_combo.currentIndexChanged.connect(self._on_planner_changed)
        self.reset_button.clicked.connect(self.parameters.reset)
        self.video.toggled.connect(self.video_length.setEnabled)

        for widget in (self.exp_num, self.num_envs, self.video_length):
            widget.valueChanged.connect(self._update_command)
        for widget in (self.headless, self.log, self.follow_camera, self.video):
            widget.toggled.connect(self._update_command)

        self.copy_button.clicked.connect(lambda: QApplication.clipboard().setText(self.command.text()))
        self.clear_button.clicked.connect(self.console.clear)
        self.start_button.clicked.connect(self.start)
        self.stop_button.clicked.connect(self.runner.stop)
        QShortcut(QKeySequence("Ctrl+Return"), self, activated=self.start)

        self.runner.output.connect(self._append_output)
        self.runner.started.connect(self._on_started)
        self.runner.finished.connect(self._on_finished)

    def _selection(self):
        problem_id, scenario_id = self.scenario_combo.currentData()
        return problem_id, scenario_id, self.planner_combo.currentData()

    def _run_options(self):
        problem_id, scenario_id, _ = self._selection()
        return {
            "task": problem_id,
            "space": scenario_id,
            "exp_num": self.exp_num.value(),
            "num_envs": self.num_envs.value(),
            "headless": self.headless.isChecked(),
            "video": self.video.isChecked(),
            "video_length": self.video_length.value(),
            "log": self.log.isChecked(),
            "follow_camera": self.follow_camera.isChecked(),
        }

    def _on_scenario_changed(self):
        _, scenario_id, _ = self._selection()
        compatible = self.planners.for_space(scenario_id)

        self.planner_combo.blockSignals(True)
        self.planner_combo.clear()
        for planner_id, planner in compatible.items():
            self.planner_combo.addItem(planner["name"], userData=planner_id)
        self.planner_combo.blockSignals(False)
        self._on_planner_changed()

    def _on_planner_changed(self):
        planner_id = self.planner_combo.currentData()
        parameters = self.planners.items[planner_id]["parameters"] if planner_id else {}
        self.parameters.set_parameters(parameters)
        self.start_button.setEnabled(planner_id is not None and not self.runner.is_running())
        self._update_command()

    def _update_command(self):
        if self.planner_combo.currentData() is None:
            self.command.setText("No planner is registered for this scenario.")
            return
        self.command.setText(shlex.join(["python"] + build_command(self._run_options())))
        self.command.setCursorPosition(0)

    def start(self):
        _, _, planner_id = self._selection()
        if planner_id is None or self.runner.is_running():
            return
        config_path = write_agent_config(planner_id, self.parameters.values())
        args = build_command(self._run_options())
        self._append_output(f"$ {shlex.join(['python'] + args)}\n")
        self._append_output(f"  planner configuration exported to {config_path}\n\n")
        self.runner.start(args, config_path)

    def _on_started(self):
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self._set_status(f"Running  ·  PID {self.runner.process.processId()}", "running")

    def _on_finished(self, code):
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self._append_output(f"\nProcess finished with exit code {code}.\n")
        self._set_status(f"Finished  ·  exit code {code}", "ok" if code == 0 else "error")

    def _set_status(self, text, state):
        self.status.set_state(text, state)

    def _append_output(self, text):
        self.console.moveCursor(QTextCursor.MoveOperation.End)
        self.console.insertPlainText(text)
        self.console.moveCursor(QTextCursor.MoveOperation.End)

    def closeEvent(self, event):
        if self.runner.is_running():
            answer = QMessageBox.question(
                self, "Experiment running",
                "An experiment is still running. Stop it and exit?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                event.ignore()
                return
            self.runner.stop()
        event.accept()


def load_theme():
    return THEME.read_text(encoding="utf-8").replace("{icons}", ICONS.as_posix())


def apply_palette(app):
    """Dark palette for the parts Fusion draws itself (menus, dialogs, disabled text)."""
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor("#0E1116"))
    palette.setColor(QPalette.Base, QColor("#1D222B"))
    palette.setColor(QPalette.AlternateBase, QColor("#161A21"))
    palette.setColor(QPalette.Text, QColor("#E6E9EF"))
    palette.setColor(QPalette.WindowText, QColor("#E6E9EF"))
    palette.setColor(QPalette.Button, QColor("#1D222B"))
    palette.setColor(QPalette.ButtonText, QColor("#E6E9EF"))
    palette.setColor(QPalette.Highlight, QColor("#76B900"))
    palette.setColor(QPalette.HighlightedText, QColor("#0B0F0A"))
    palette.setColor(QPalette.ToolTipBase, QColor("#1D222B"))
    palette.setColor(QPalette.ToolTipText, QColor("#E6E9EF"))
    palette.setColor(QPalette.Disabled, QPalette.Text, QColor("#5B6370"))
    palette.setColor(QPalette.Disabled, QPalette.WindowText, QColor("#5B6370"))
    palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor("#5B6370"))
    app.setPalette(palette)


def main():
    QLocale.setDefault(QLocale.c())
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    apply_palette(app)
    app.setStyleSheet(load_theme())
    window = MainWindow(Problems(), Planners())
    window.show()
    sys.exit(app.exec())
