"""Reading the keyboard and writing to the terminal, without the viewer in the way."""

from __future__ import annotations

import os
import select
import sys

class Console:
    """Keeps the commands on one line and everything else on its own.

    The commands change constantly while driving, so they are rewritten in place instead
    of scrolling the window; a state change, which is worth keeping, gets a line of its own.
    """

    def __init__(self):
        self.width = 0
        self.inline = False

    def line(self, text: str):
        """A message that stays on the screen."""
        if self.inline:
            print(flush=True)       # finish the line the commands are on
            self.inline = False
        print(text, flush=True)

    def inplace(self, text: str):
        """The commands, rewritten over the line they are already on."""
        self.width = max(self.width, len(text))
        print("\r" + text.ljust(self.width), end="", flush=True)
        self.inline = True

    def close(self):
        if self.inline:
            print(flush=True)
            self.inline = False


class TerminalKeys:
    """Keys read from the terminal, which the viewer never sees.

    The viewer owns almost every letter for its own render toggles -- W wireframe, T
    transparent, F contact forces -- so driving the robot in its window flips those on and
    off instead. Reading the terminal keeps the two apart: the viewer shows the robot, the
    terminal drives it. Keep the terminal focused while driving.

    Keys arrive as the same codes the viewer would have reported, so the key map does not
    care where they came from, and a key held down repeats here as it does anywhere else.
    """

    ARROWS = {"A": 265, "B": 264, "C": 262, "D": 263}   # up, down, right, left

    def __init__(self):
        self.fd = sys.stdin.fileno()
        self.saved = None

    @property
    def usable(self) -> bool:
        return sys.stdin.isatty()

    def __enter__(self):
        if self.usable:
            import termios
            import tty

            self.saved = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)      # keys arrive one at a time, and Ctrl-C still works
        return self

    def __exit__(self, *exc):
        if self.saved is not None:
            import termios

            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.saved)
            self.saved = None

    def _read(self) -> str:
        return os.read(self.fd, 1).decode(errors="ignore")

    def _waiting(self, timeout: float = 0.0) -> bool:
        return bool(select.select([self.fd], [], [], timeout)[0])

    def poll(self) -> list[int]:
        """Every key pressed since the last call."""
        keys = []
        while self.saved is not None and self._waiting():
            char = self._read()
            if not char:
                break
            if char == "\x1b":
                # an escape of its own, or the start of an arrow key: ESC [ A..D
                sequence = ""
                while len(sequence) < 2 and self._waiting(0.001):
                    sequence += self._read()
                keys.append(self.ARROWS.get(sequence[1:2], 256) if sequence.startswith("[") else 256)
            elif char in ("\x03", "\x04"):      # Ctrl-C, Ctrl-D
                keys.append(256)
            else:
                keys.append(ord(char.upper()))
        return keys
