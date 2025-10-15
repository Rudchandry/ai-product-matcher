# Redirect to root index.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from index import handler as root_handler

def handler(event, context):
    return root_handler(event, context)