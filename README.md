# Pixelated Evolution

![](https://github.com/pwspen/pterobot/pixevo.gif)

This is a project, written in Python, that simulates evolution. I like to think of it like "evolution where pixels are the unit of information".

Above, you can watch the right side of the image, starting from random noise, evolve to match the left side.

The end goal of this project is to create an environment in which to examine the process of evolution and run tests on it. I'm also going to create a web interface intended to teach evolution in a visual and easy-to-understand way: as the image evolves visually, the evolutionary family tree will be generated alongside it.

The core loop is: Parent image creates child images with mutations. Child images closer to the target image (higher fitness) have more children of their own. That's it.

There are a number of different parameters and modes that can be set that shape evolution: generation size, children distribution as function of parent fitness, mutation fraction (can be set as function of fitness), etc.

Evolution is interesting to me because of how it contrasts with machine learning algorithms: it is totally 'unguided' which has advantages and disadvantages, and it is much simpler. 
