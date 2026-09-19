"""Reusable widgets for the launcher."""

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


def field_label(text):
    label = QLabel(text)
    label.setObjectName("fieldLabel")
    return label


def humanize(name):
    """``max_depth`` -> ``Max depth``; short symbolic names such as ``k`` are kept."""
    if len(name) <= 2:
        return name
    return name.replace("_", " ").capitalize()


def expanding(widget):
    """Let an input fill the field column of a form."""
    widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    return widget


class Section(QFrame):
    """Card with an upper-case header, an optional header action and a body layout."""

    def __init__(self, title, action=None, parent=None):
        super().__init__(parent)
        self.setObjectName("card")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 14, 16, 16)
        outer.setSpacing(12)

        header = QHBoxLayout()
        heading = QLabel(title.upper())
        heading.setObjectName("sectionTitle")
        font = QFont(heading.font())
        font.setLetterSpacing(QFont.PercentageSpacing, 108)
        heading.setFont(font)
        header.addWidget(heading)
        header.addStretch()
        if action is not None:
            header.addWidget(action)
        outer.addLayout(header)

        self.body = QVBoxLayout()
        self.body.setSpacing(10)
        outer.addLayout(self.body)

    def add_form(self):
        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setHorizontalSpacing(14)
        form.setVerticalSpacing(10)
        form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        self.body.addLayout(form)
        return form


class StatusPill(QFrame):
    """Colored dot plus text; the dot color follows the ``state`` property (see theme.qss)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("status")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 4, 14, 4)
        layout.setSpacing(8)
        self.dot = QLabel("\u25cf")
        self.dot.setObjectName("statusDot")
        self.text = QLabel()
        self.text.setObjectName("statusText")
        layout.addWidget(self.dot)
        layout.addWidget(self.text)

    def set_state(self, text, state):
        self.text.setText(text)
        self.dot.setProperty("state", state)
        self.dot.style().unpolish(self.dot)
        self.dot.style().polish(self.dot)


class ParameterForm(QWidget):
    """Form generated from ``{name: default}``; the widget type follows the default's type."""

    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._layout = QFormLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setHorizontalSpacing(14)
        self._layout.setVerticalSpacing(10)
        self._layout.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        self._widgets = {}
        self._defaults = {}

    def set_parameters(self, parameters):
        while self._layout.rowCount():
            self._layout.removeRow(0)
        self._widgets.clear()
        self._defaults = dict(parameters)

        if not parameters:
            note = QLabel("This planner has no configurable parameters.")
            note.setObjectName("hint")
            self._layout.addRow(note)
            return

        for name, default in parameters.items():
            widget = expanding(self._make_widget(default))
            widget.setToolTip(name)
            self._widgets[name] = widget
            self._layout.addRow(field_label(humanize(name)), widget)

    def reset(self):
        for name, widget in self._widgets.items():
            self._set_value(widget, self._defaults[name])

    def values(self):
        return {name: self._value(widget) for name, widget in self._widgets.items()}

    # ------------------------------------------------------------------

    def _make_widget(self, default):
        if isinstance(default, bool):
            widget = QCheckBox("Enabled")
            widget.setChecked(default)
            widget.toggled.connect(self.changed)
        elif isinstance(default, int):
            widget = QSpinBox()
            widget.setRange(0, 1_000_000)
            widget.setValue(default)
            widget.valueChanged.connect(self.changed)
        elif isinstance(default, float):
            widget = QDoubleSpinBox()
            widget.setRange(0.0, 1_000_000.0)
            widget.setDecimals(4)
            widget.setSingleStep(0.01)
            widget.setValue(default)
            widget.valueChanged.connect(self.changed)
        else:
            widget = QLineEdit(str(default))
            widget.textChanged.connect(self.changed)
        return widget

    @staticmethod
    def _value(widget):
        if isinstance(widget, QCheckBox):
            return widget.isChecked()
        if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
            return widget.value()
        return widget.text()

    @staticmethod
    def _set_value(widget, value):
        if isinstance(widget, QCheckBox):
            widget.setChecked(value)
        elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
            widget.setValue(value)
        else:
            widget.setText(str(value))
