"""A driving session: the robot, the viewer, the keyboard and the clock."""

from __future__ import annotations

import time

from .controller import Controller
from .keyboard import Console, TerminalKeys
from .teleop import HELP, Teleop

def teleoperate(sim: "Sim2Sim", height: float, mode: str = "sticky", keys: str = "terminal",
                speed: float = 1.0):
    import mujoco.viewer

    teleop = Teleop(sim.cfg, height, mode)
    controller = Controller(sim)
    controller.speed = speed
    sim.lie_down()
    dt = sim.model.opt.timestep * sim.cfg["decimation"]
    terminal = TerminalKeys()
    if keys == "terminal" and not terminal.usable:
        print("  this is not a terminal: reading the keys from the viewer window instead")
        keys = "viewer"
    print(HELP)
    print(f"  driving: {mode}   keys: {keys}   "
          f"up to {'/'.join(f'{l:.2g}' for l in controller.command_limits)} (vx, vy, wz)")
    if sim.policy is None:
        print("  no policy loaded: press 2 to stand the robot up, then 6 to walk it\n")
    else:
        print("  the robot starts lying down: press 2 to stand it up, then 3 for the policy"
              " or 6 to walk without one\n")

    console = Console()
    callback = teleop.key_callback if keys == "viewer" else None
    with terminal, mujoco.viewer.launch_passive(sim.model, sim.data, key_callback=callback) as viewer:
        next_frame = time.time()
        last_banner = 0.0
        while viewer.is_running() and not teleop.stop:
            if keys == "terminal":
                for key in terminal.poll():
                    teleop.key_callback(key)
            if teleop.reset:
                sim.lie_down()
                controller.__init__(sim)
                teleop.posture[:] = 0.0
                teleop.reset = False
                console.line("state: damping (robot reset)")
            if teleop.requested_state is not None:
                wanted = teleop.requested_state
                if wanted == controller.state and wanted in controller.DRIVEN:
                    wanted = controller.STAND   # asking again hands the legs back
                console.line(controller.request(wanted))
                teleop.posture[:] = controller.posture_target   # the controller may overrule it
                teleop.requested_state = None
            teleop.limits = controller.command_limits
            teleop.update(dt)
            if teleop.changed and time.time() - last_banner > 0.1:
                if controller.state in controller.DRIVEN or controller.state == controller.STAND:
                    console.inplace(teleop.banner())
                teleop.changed = False
                last_banner = time.time()

            # the keys move the trunk only while the controller is holding the robot for
            # its own sake; once the legs are being handed over they belong to whoever is
            # about to drive them, and a posture command would undo the squaring up
            handing_over = controller.pending is not None
            teleop.posture_mode = controller.state == controller.STAND and not handing_over
            if handing_over:
                teleop.posture[:] = 0.0
            controller.posture_target = teleop.posture
            sim.height_command = teleop.height
            message = controller.update(teleop.command)
            if message:
                console.line(message)
            viewer.sync()

            # keep the simulation in step with the wall clock, as the robot would be
            next_frame += dt
            time.sleep(max(0.0, next_frame - time.time()))
        console.close()
