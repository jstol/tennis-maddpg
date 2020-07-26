#!/usr/bin/env python
# coding: utf-8
"""Main entry point – uses MADDPG to solve the "Tennis" Unity ML-Agents environment"""

# Third party imports
import click

# Project imports
from maddpg import train_model, view_agent


# Define CLI
@click.group(help='train/visualize an MADDPG agent within the Unity ml-agents "Tennis" environment')
def cli():
    pass


@cli.command(help='Train an MADDPG model')
def train():
    train_model()


@cli.command(help='View pre-trained agents, saved under POLICY_PATH')
@click.argument('policy-path', type=str)
def view(policy_path):
    view_agent(policy_path)


if __name__ == '__main__':
    cli()
