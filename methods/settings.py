import datetime
import os

"""
This file contains global setting for the project
and oher utility functions
"""

# global settings for the project
pwd = os.path.dirname(os.path.abspath(__file__))

# Determine the hostname
import socket
hostname = socket.gethostname()

settings = {
    # Paths
    'project_directory': pwd,
    'data_directory': os.path.abspath(os.path.join(pwd, '../../data/')),
    'output_directory': os.path.abspath(os.path.join(pwd, '../output/')),
    # Hostname
    'hostname': hostname,
    # Project settings
    'today': datetime.date.today(),
    'authors': 'Aaron Banks',
    # register print host, managed by main
    'print_host': False,
}

