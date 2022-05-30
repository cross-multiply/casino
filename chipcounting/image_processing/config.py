import threading

cv_local_config = threading.local()


def reinit_local():
    global cv_local_config
    cv_local_config = threading.local()


def get_local_config():
    global cv_local_config
    return cv_local_config
