import time
import mujoco
import mujoco.viewer

### Usage:
# Open XML in viewier with this file, then from viewier save MjData,
# then use update_posed_from_mjdata.py to scan qp and qv -> updates env.py
# Todo: 
# convert to using keyframes instead of pasting arrays

m = mujoco.MjModel.from_xml_path('load/pterobot_v0.xml')

# sim = mujoco.Simulation(m)
viewer = mujoco.viewer.launch(m)
# while True:
#     mujoco.mj_step(m, viewer.data)
#     viewer.render()
# m = mujoco.MjModel.from_xml_path('posed.xml')
# m = mujoco.mj_loadModel('mjmodel.mjb')
# m = mujoco.MjModel.loadModel('mjmodel.mjb')
# m = mujoco.MjModel.('mjmodel.mjb')

# d = mujoco.MjData(m)

# # mujoco.mj_saveLastXML('BA14/urdf/Finally.xml', m)


# with mujoco.viewer.launch(m, d) as viewer:
#   # Close the viewer automatically after 30 wall-seconds.
#   start = time.time()
#   while viewer.is_running():
#     step_start = time.time()

#     # mj_step can be replaced with code that also evaluates
#     # a policy and applies a control signal before stepping the physics.
#     mujoco.mj_step(m, d)

#     # Example modification of a viewer option: toggle contact points every two seconds.
#     with viewer.lock():
#       viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)

#     # Pick up changes to the physics state, apply perturbations, update options from GUI.
#     viewer.sync()

#     # Rudimentary time keeping, will drift relative to wall clock.
#     time_until_next_step = m.opt.timestep - (time.time() - step_start)
#     if time_until_next_step > 0:
#       time.sleep(time_until_next_step)