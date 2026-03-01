import numpy as np

times = 4
forces = [
    {  # 
        'step': 5,
        'end_step': 6,
        'force': f,
        'times': times,
        'direction': np.array([0,1,0]),
        'success': [False]*times
    } for f in [40.0, 45.0, 50.0, 55.0]] + [
    { # 
        'step': 5,
        'end_step': 6,
        'force': f,
        'times': times,
        'direction': np.array([0,-0,0]),
        'success': [False]*times
    } for f in [40.0, 45.0, 50.0, 55.0]] +[
    { #
        'step': 5,
        'end_step': 6,
        'force': f,
        'times': times,
        'direction': np.array([1,0,0]),
        'success': [False]*times
    } for f in [40.0, 45.0, 50.0, 55.0]] +[
    {#
        'step': 5,
        'end_step': 6,
        'force': f,
        'times': times,
        'direction': np.array([-1,0,0]),
        'success': [False]*times
    } for f in [40.0, 45.0, 50.0, 55.0]]