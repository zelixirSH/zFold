"""Jizhi-related utility functions."""

import os
import json
import socket
import logging
import traceback

import requests


def get_ip():
    """Get the current node's IP.

    Args: n/a

    Returns:
    * ip_addr: current node's IP
    """

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(('10.223.30.51', 53))
        ip_addr = sock.getsockname()[0]
    except Exception as err:  # pylint: disable=broad-except
        ip_addr = socket.gethostbyname(socket.gethostname())
        logging.warning('failed to get the current node\'s IP: %s', err)
    finally:
        sock.close()

    return ip_addr


def report_progress(msg):
    """Report the training progress message.

    Args:
    * msg: (key, value) pairs of critical indexes for the training progress

    Returns:
    * flag: whether the training progress message has been successfully reported
    * text: server's response text
    """

    ip_addr = os.environ.get('CHIEF_IP', '')
    if not ip_addr:
        ip_addr = get_ip()
    url = 'http://%s:%s/v1/worker/report-progress' % (ip_addr, 8080)
    err_frmt = 'send progress info to worker failed!\nprogress_info: %s, \n%s: %s'

    try:
        response = requests.post(url, json=json.dumps(msg), proxies={"http": None, "https": None})
    except Exception as err:  # pylint: disable=broad-except
        logging.warning(err_frmt, msg, 'traceback', traceback.format_exc())
        return False, str(err)

    if response.status_code != 200:
        logging.warning(err_frmt, msg, 'reason', response.reason)
        return False, response.text

    return True, ''


def report_error(code, msg=''):
    """Report the error message.

    Args:
    * code: error code
    * msg: error message

    Returns:
    * flag: whether the error message has been successfully reported
    * text: server's response text
    """

    err_msg = {'type': 'error', 'code': code, 'msg': msg}

    return report_progress(err_msg)


def report_completion():
    """Report the task completion message.

    Args: n/a

    Returns:
    * flag: whether the task completion message has been successfully reported
    * text: server's response text
    """

    msg = {'type': 'completed'}

    return report_progress(msg)
