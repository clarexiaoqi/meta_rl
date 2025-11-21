# Meta-Learning–Based Controller for Thermal Management (TensorFlow 2 Version)
## Overview

This repository contains the TensorFlow 2–refactored version of the original Meta-RL temperature-control framework. The goal of this project is to develop an autonomous controller for a single office space conditioned by a VAV HVAC system. Using a meta-reinforcement learning approach, the controller learns an adaptive policy that can generalize across different thermal scenarios.

The agent regulates heating and cooling inputs based on current building states—such as zone temperature, internal heat gains, and solar gains. The learned control policy aims to balance occupant comfort and energy efficiency, and is optimized using a meta-RL algorithm to maximize long-term returns.

A detailed explanation of the original methodology, system modeling, and algorithm design can be found here:

https://uofnebraska-my.sharepoint.com/:f:/g/personal/81973916_nebraska_edu/Ekz5l_tdAD5GqCXL0HO-8awB8ew56lqCUiYN2_B7xWiCsw?e=nh7cgy
