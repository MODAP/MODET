NOTE: there is pretty much nothing in the master branch because.... We have no release code yet. See other branches for better fun!

![Image](https://raw.githubusercontent.com/MODAP/MODET/master/LOGO.png?raw=true)
# Motorized Object DETection
An Open-Source Detection Module

# What?
We are MODAP, a not-for-profit organization that creates innovative search and rescue solutions for California's wildfires.

This module is aiming to use image recognition technologies and practices such as RNNs, SqueezeNets, etc. to create a training, detection, and deployment all-in-one solution.

# Who?
The team at MODAP will be hard at work developing this module. If you wish to help out, we welcome all pulls and review them carefully. See the [contributing](#contributing) section for info.

# How?
To use this package, clone it. There are three sub-packages in the `MODET` package: `brain`, `love`, and `god`. The front end developer should only need to interface with `MODET.god` as it is the primary training, testing, and prediction handler. 

`MODET.brain` provides the ML Keras graphs. There is usually no need to interact with it unless you are implementing custom modules.

`MODET.love` provides import-export handling. In addition to being used in exporting and deploying trained graphs, the package also handles information transferred between `MODET.brain` and `MODET.god`.

Please see [the wiki](https://github.com/MODAP/MODET/wiki) for usage and documentation. 


# Contributing
Thank you so much for your interest in MODET. We are a small team working often remotely, so it is always helpful when we have an extra hand.

[Our wiki](https://github.com/MODAP/MODET/wiki) provides perhaps the best amount of information with regards to our current goals and timelines. Please be sure to review it so that your contribution is not falling behind.

When creating a pull request, consider the following:
* Is my code neat and understandable? (i.e. well commented, following best practices, etc.)
* Can the developers of MODET extend the feature/fix easily?
* Am I following the code of conduct?

Please do not hesitate to ask any and all questions to the developers via Github's many methods or [via GITTER](https://gitter.im/MODAP/community).

