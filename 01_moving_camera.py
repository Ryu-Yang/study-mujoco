"""
MuJoCo 动态相机演示：模拟多米诺骨牌效应并实现相机平滑移动

本脚本演示了如何使用 MuJoCo 物理引擎创建一个包含多个刚体的场景，
并实现相机从轨道视角平滑过渡到跟踪特定物体的效果。
"""

import mujoco  # MuJoCo 物理引擎主库


# 导入
import numpy as np  # 数值计算库

import mediapy as media  # 视频处理库
import matplotlib.pyplot as plt  # 绘图库

import cv2  # OpenCV 计算机视觉库

# 设置 numpy 打印选项，使输出更易读
np.set_printoptions(precision=3, suppress=True, linewidth=100)


# ============================================================================
# MuJoCo XML 模型定义（模块化设计）
# ============================================================================
"""
定义了一个包含多个刚体的物理场景，采用模块化设计：
- 环境设置（地板、斜坡、光源、相机）在 environment.xml
- 球体和盒子在 ball_and_box.xml
- 6个多米诺骨牌在 domino_chain.xml
- 跷跷板结构在 seesaw.xml
- 主文件 main.xml 使用 <include> 指令整合所有组件
"""

# XML 文件路径
xml_path = "descriptions/dominos_scene/main.xml"

# ============================================================================
# 模型和数据初始化
# ============================================================================

# 从 XML 文件创建 MuJoCo 模型
model = mujoco.MjModel.from_xml_path(xml_path)

print("模型统计信息:")
print(f"nq (广义坐标): {model.nq}")
print(f"nv (自由度): {model.nv}") 
print(f"nbody (刚体): {model.nbody}")
print(f"ngeom (几何体): {model.ngeom}")
print(f"njnt (关节): {model.njnt}")


# 创建与模型关联的仿真数据
data = mujoco.MjData(model)

# 仿真参数
duration = 3  # 仿真时长（秒）
height = 1024  # 渲染高度（像素）
width = 1440   # 渲染宽度（像素）

# ============================================================================
# 检测盒子被抛出的时刻
# ============================================================================
"""
运行初始仿真以确定盒子何时被撞击（速度超过 2cm/s）。
这个时间点将用于相机运动的过渡。
"""
throw_time = 0.0  # 盒子被抛出的时间
mujoco.mj_resetData(model, data)  # 重置仿真数据

# 运行仿真直到找到抛出时间或达到总时长
while data.time < duration and not throw_time:
    mujoco.mj_step(model, data)  # 执行一个仿真步长
    
    # 计算盒子的线速度（前3个自由度）
    box_speed = np.linalg.norm(data.joint('box').qvel[:3])
    
    # 如果速度超过阈值（2cm/s），记录抛出时间
    if box_speed > 0.02:
        throw_time = data.time

# 确保找到了抛出时间
assert throw_time > 0, "未检测到盒子被抛出！"

# ============================================================================
# 相机运动辅助函数
# ============================================================================

def mix(time, t0=0.0, width=1.0):
    """
    Sigmoid 混合函数：计算两个权重 w0 和 w1，用于平滑过渡。
    
    参数：
        time: 当前时间
        t0: 过渡开始时间
        width: 过渡宽度（控制过渡速度）
    
    返回：
        w0, w1: 两个权重，满足 w0 + w1 = 1
    """
    t = (time - t0) / width
    s = 1 / (1 + np.exp(-t))  # Sigmoid 函数
    return 1 - s, s  # w0 从 1 到 0，w1 从 0 到 1


def unit_cos(t):
    """
    单位余弦 Sigmoid：将输入从 [0,1] 映射到 [0,1] 的平滑函数。
    
    参数：
        t: 输入值（被限制在 [0,1] 范围内）
    
    返回：
        平滑过渡的值，从 (0,0) 到 (1,1)
    """
    return 0.5 - np.cos(np.pi * np.clip(t, 0, 1)) / 2


def orbit_motion(t):
    """
    轨道运动：相机围绕场景的轨道运动。
    
    参数：
        t: 归一化时间（0 到 1）
    
    返回：
        distance: 相机距离
        azimuth: 方位角（度）
        elevation: 仰角（度）
        lookat: 观察点坐标
    """
    distance = 0.9  # 固定距离
    azimuth = 140 + 100 * unit_cos(t)  # 方位角从 140 度平滑变化到 240 度
    elevation = -30  # 固定仰角
    lookat = data.geom('floor').xpos.copy()  # 观察点在地板中心
    return distance, azimuth, elevation, lookat


def track_motion():
    """
    跟踪运动：相机跟踪盒子物体的运动。
    
    返回：
        distance: 相机距离（较近）
        azimuth: 方位角（固定）
        elevation: 仰角（固定）
        lookat: 观察点（盒子当前位置）
    """
    distance = 0.08  # 近距离跟踪
    azimuth = 280  # 固定方位角
    elevation = -10  # 固定仰角
    lookat = data.geom('box').xpos.copy()  # 观察点跟随盒子
    return distance, azimuth, elevation, lookat


def cam_motion():
    """
    相机运动：结合轨道运动和跟踪运动的混合相机轨迹。
    
    使用 Sigmoid 混合函数在抛出时间附近平滑过渡：
    - 抛出前：主要使用轨道运动
    - 抛出后：逐渐过渡到跟踪运动
    
    返回：
        混合后的相机参数
    """
    # 计算轨道运动参数（使用归一化时间）
    d0, a0, e0, l0 = orbit_motion(data.time / throw_time)
    
    # 计算跟踪运动参数
    d1, a1, e1, l1 = track_motion()
    
    # 混合参数：在抛出时间附近进行 0.3 秒的过渡
    mix_time = 0.3
    w0, w1 = mix(data.time, throw_time, mix_time)
    
    # 线性混合两个轨迹
    return (w0 * d0 + w1 * d1,
            w0 * a0 + w1 * a1,
            w0 * e0 + w1 * e1,
            w0 * l0 + w1 * l1)


# ============================================================================
# 相机设置
# ============================================================================

# 创建相机对象
cam = mujoco.MjvCamera()
# 设置相机默认参数
mujoco.mjv_defaultCamera(cam)

# ============================================================================
# 主仿真循环：渲染视频
# ============================================================================

framerate = 60  # 帧率（Hz）
slowdown = 4    # 慢放倍数（4倍慢放）

# 重置仿真数据
mujoco.mj_resetData(model, data)
frames = []  # 存储渲染帧

# 使用渲染器上下文管理器
with mujoco.Renderer(model, height, width) as renderer:
    # 运行仿真循环
    while data.time < duration:
        # 执行一个仿真步长
        mujoco.mj_step(model, data)
        
        # 根据慢放倍数控制帧采集
        if len(frames) < data.time * framerate * slowdown:
            # 更新相机参数
            cam.distance, cam.azimuth, cam.elevation, cam.lookat = cam_motion()
            
            # 更新渲染场景
            renderer.update_scene(data, cam)
            
            # 渲染当前帧
            pixels = renderer.render()
            
            # 使用 OpenCV 显示实时渲染（BGR 格式）
            cv2.imshow("render", cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)  # 短暂等待以显示图像
            
            # 保存帧到列表
            frames.append(pixels)

# ============================================================================
# 显示最终视频
# ============================================================================

# 使用 mediapy 显示生成的视频
media.show_video(frames, fps=framerate)

"""
脚本执行流程总结：
1. 定义物理场景（多米诺骨牌、球体、盒子等）
2. 初始化 MuJoCo 模型和仿真数据
3. 预运行仿真以检测盒子被抛出的时间
4. 定义相机运动轨迹（轨道运动 + 跟踪运动）
5. 运行主仿真循环，实时渲染并应用动态相机
6. 显示最终生成的视频

关键特性：
- 物理精确的多米诺骨牌效应仿真
- 相机平滑过渡：从全局视角到特写跟踪
- 实时渲染显示
- 4倍慢放效果，便于观察细节
"""
