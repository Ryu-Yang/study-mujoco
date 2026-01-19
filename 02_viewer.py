import mujoco
import mujoco.viewer
import time

model = mujoco.MjModel.from_xml_path('mujoco_menagerie/agilex_piper/scene.xml')
data = mujoco.MjData(model)
paused = False

def key_callback(keycode):
  if chr(keycode) == ' ':
    global paused
    paused = not paused

with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
    while viewer.is_running():
        if not paused:
            mujoco.mj_step(model,data)
            viewer.sync()
        time.sleep(0.002)   #合适的延迟控制render的速率