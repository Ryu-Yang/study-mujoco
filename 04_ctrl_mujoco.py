import numpy as np
import mujoco
import mujoco.viewer
import time


########################## 加载模型 ##########################
model = mujoco.MjModel.from_xml_path('mujoco_menagerie/franka_emika_panda/scene.xml')
data = mujoco.MjData(model)

# 设置仿真时间步长
model.opt.timestep = 0.01

print("模型信息:")
print(f"自由度 (nv): {model.nv}")
print(f"关节数量 (njnt): {model.njnt}")
print(f"执行器数量 (nu): {model.nu}")


########################## 获取执行器索引 ##########################
# 对应的执行器名称（从 xml 中获取）
actuator_names = [
    'actuator1',
    'actuator2',
    'actuator3',
    'actuator4',
    'actuator5',
    'actuator6',
    'actuator7',
    'actuator8',
]

actuators_idx = []
for _, actuator_name in enumerate(actuator_names):
    actuators_idx.append({
        "id": model.actuator(actuator_name).id,
        "name": actuator_name
    })

print(f"找到 {len(actuators_idx)} 个执行器")
for dof in actuators_idx:
    print(f" 执行器 '{dof['name']}' (ID: {dof['id']})")


########################## 控制参数 ##########################
def finger_position_to_control(pos_meters):
    """将手指位置（米）转换为控制信号（0-255）"""
    return np.clip(pos_meters * 255 / 0.04, 0, 255)


########################## 仿真循环 ##########################
mujoco.mj_resetData(model, data) # 重置仿真数据

try:
    viewer = mujoco.viewer.launch_passive(model, data)
    viewer_available = True
except:
    print("Viewer 不可用，将在无可视化模式下运行")
    viewer_available = False

print("开始仿真...")
print("按 Ctrl+C 停止")

try:
    while True:
        # current_mode = mode
        target = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        current_target = target.copy()
        if target is not None:
            
            # 转换手指位置为控制信号
            if len(current_target) >= 8:
                current_target[7] = finger_position_to_control(current_target[7])

            for j, dof in enumerate(actuators_idx):
                if dof['id'] >= 0 and j < len(current_target):
                    data.ctrl[dof['id']] = current_target[j]
                    
        # 执行仿真步长
        mujoco.mj_step(model, data)
        
        # 更新 viewer
        if viewer_available:
            viewer.sync()
            # 添加小延迟以控制仿真速度
            time.sleep(0.002)
        
            
except KeyboardInterrupt:
    print("\n仿真被用户中断")
finally:
    if viewer_available:
        viewer.close()
    
print("仿真结束")
