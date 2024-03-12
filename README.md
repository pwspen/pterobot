# Pixelated Evolution

<<<<<<< HEAD
![](https://github.com/pwspen/pterobot/stumble.gif)

![](https://github.com/pwspen/pterobot/cad.png)

Pterosaurs were the membrane-winged flying animals that lived alongside the dinosaurs. Their locomotion (walks on 4 legs, flighted) is very interesting, so I want to make a robot that walks (and maybe eventually flies) in the same way.

I'm starting the project with the motion control part. I'm using reinforcement learning to train a neural network to walk.

I'm using MuJoCo as a physics simulator, Brax for machine learning policies + training, and Jax for runtime optimization. All three are developed by Google Deepmind, and I'm also using google GPUs to train, through google colab.

I haven't quite got it walking yet (as you can see), but I'm getting close. Once I do get it walking semi-gracefully, I can start thinking more about the mechatronic part of this project.

Developing a walking robot will be difficult, and developing a flapping + flying robot will be almost impossible. Thus, this is a pretty long-term project.
=======
![](https://github.com/pwspen/pterobot/pixevo.gif)

This is a project, written in Python, that simulates evolution. I like to think of it like "evolution where pixels are the unit of information".

Above, you can watch the right side of the image, starting from random noise, evolve to match the left side.

The end goal of this project is to create an environment in which to examine the process of evolution and run tests on it. I'm also going to create a web interface intended to teach evolution in a visual and easy-to-understand way: as the image evolves visually, the evolutionary family tree will be generated alongside it.

The core loop is: Parent image creates child images with mutations. Child images closer to the target image (higher fitness) have more children of their own. That's it.

There are a number of different parameters and modes that can be set that shape evolution: generation size, children distribution as function of parent fitness, mutation fraction (can be set as function of fitness), etc.

Evolution is interesting to me because of how it contrasts with machine learning algorithms: it is totally 'unguided' which has advantages and disadvantages, and it is much simpler. 
>>>>>>> 353759cd771c627a3a67216167e36511e01e1efa
