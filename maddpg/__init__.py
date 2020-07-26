#!/usr/bin/env python
# coding: utf-8
"""An implementation of MADDPG to solve the "Tennis" Unity ML-Agents environment."""

from .runner import train_model, view_agent, ReplayBuffer
from .models import QNet, PolicyNet

# Seed RNGs
import random as _random
import numpy as _np

SEED = 123
_random.seed(SEED)
_np.random.seed(SEED)


