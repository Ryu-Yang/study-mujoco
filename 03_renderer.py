import mujoco
import mujoco.viewer
import time
import cv2

model = mujoco.MjModel.from_xml_path('mujoco_menagerie/robotstudio_so101/scene.xml')
data = mujoco.MjData(model)
paused = False

height = 480  # 渲染高度（像素）
width = 640   # 渲染宽度（像素）

def key_callback(keycode):
  if chr(keycode) == ' ':
    global paused
    paused = not paused

# 创建相机对象
cam = mujoco.MjvCamera()

# 设置相机默认参数
mujoco.mjv_defaultCamera(cam)

mujoco.mj_resetData(model, data)

with mujoco.Renderer(model, height, width) as renderer:
    while True:
        mujoco.mj_step(model, data)
        
        # 更新渲染场景
        renderer.update_scene(data, cam)
            
        # 渲染当前帧
        pixels = renderer.render()
            
        # 使用 OpenCV 显示实时渲染（BGR 格式）
        cv2.imshow("render", cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)  # 短暂等待以显示图像

        time.sleep(0.03)   #合适的延迟控制render的速率
