import mujoco
import os

# Path to your URDF file
urdf_file_path = 'path/to/your/robot.urdf'

# Load the URDF file directly into MuJoCo
model = mujoco.load_model_from_path(urdf_file_path)

# Create the MjSim object with the loaded model
sim = mujoco.MjSim(model)

# Create a viewer to visualize the simulation
viewer = mujoco.MjViewer(sim)

print("Use arrow keys to move the camera, ESC to exit.")
while True:
    # Step the simulation
    sim.step()

    # Render the scene
    viewer.render()

    # Check for user input to break the loop
    if viewer.exit:
        break

print("Simulation ended.")