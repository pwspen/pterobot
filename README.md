# Robotic Pterosaur

![](https://github.com/pwspen/pterobot/stumble.gif)

![](https://github.com/pwspen/pterobot/cad.png)

Pterosaurs were the membrane-winged flying animals that lived alongside the dinosaurs. Their locomotion (walks on 4 legs, flighted) is very interesting, so I want to make a robot that walks (and maybe eventually flies) in the same way.

I'm starting the project with the motion control part. I'm using reinforcement learning to train a neural network to walk.

I'm using MuJoCo as a physics simulator, Brax for machine learning policies + training, and Jax for runtime optimization. All three are developed by Google Deepmind, and I'm also using google GPUs to train, through google colab.

I haven't quite got it walking yet (as you can see), but I'm getting close. Once I do get it walking semi-gracefully, I can start thinking more about the mechatronic part of this project.

Developing a walking robot will be difficult, and developing a flapping + flying robot will be almost impossible. Thus, this is a pretty long-term project.
