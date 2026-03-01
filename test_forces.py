import numpy as np

times = 4
forces = [
    {  # forward
        'step': 5,
        'end_step': 6,
        'force': f,
        'times': times,
        'direction': np.array([1,0,0]),
        'success': [False]*times
    } for f in [30.0, 35.0, 50.0, 60.0, 70.0, 85.0]] + [
    { #backward 
        'step': 5,
        'end_step': 6,
        'force': f,
        'times': times,
        'direction': np.array([-1,0,0]),
        'success': [False]*times
    } for f in [30.0, 40.0, 50.0, 60.0, 70.0, 85.0]] +[
    { #right
        'step': 5,
        'end_step': 6,
        'force': f,
        'times': times,
        'direction': np.array([0,1,0]),
        'success': [False]*times
    } for f in [30.0, 40.0, 50.0, 60.0, 70.0, 85.0]] +[
    {#left
        'step': 5,
        'end_step': 6,
        'force': f,
        'times': times,
        'direction': np.array([0,-1,0]),
        'success': [False]*times
    } for f in [30.0, 40.0, 50.0, 60.0, 70.0, 85.0]]