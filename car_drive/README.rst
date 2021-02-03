OpenAI Gym
**********

**OpenAI Gym is a toolkit for developing and comparing reinforcement learning algorithms.** This is the ``gym`` open-source library, which gives you access to a standardized set of environments.

Installation
============

You can perform a minimal install of ``gym`` with:

.. code:: shell

    git clone https://github.com/sixauto/conduziadrive.git
    cd car_drive
    pip3 install -e .

.. code:: shell
    pip3 install stablebaselines3 tensorboard

Run the code and show the progress on tensorboard
--------------------------------------------------

.. code:: shell
    tensorboard --logdir path

Rendering on a server
---------------------

If you're trying to render video on a server, you'll need to connect a
fake display. The easiest way to do this is by running under
``xvfb-run`` (on Ubuntu, install the ``xvfb`` package):

.. code:: shell

     xvfb-run -s "-screen 0 1400x900x24" python3 conduziadrive.py

Testing
=======

We are using `pytest <http://doc.pytest.org>`_ for tests. You can run them via:

.. code:: shell

    pytest


.. _See What's New section below:

Resources
=========

-  `OpenAI.com`_
-  `Gym.OpenAI.com`_
-  `Gym Docs`_
-  `Gym Environments`_
-  `OpenAI Twitter`_
-  `OpenAI YouTube`_

.. _OpenAI.com: https://openai.com/
.. _Gym.OpenAI.com: http://gym.openai.com/
.. _Gym Docs: http://gym.openai.com/docs/
.. _Gym Environments: http://gym.openai.com/envs/
.. _OpenAI Twitter: https://twitter.com/openai
.. _OpenAI YouTube: https://www.youtube.com/channel/UCXZCJLdBC09xxGZ6gcdrc6A
