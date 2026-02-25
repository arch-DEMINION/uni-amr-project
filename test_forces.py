import numpy as np

times = 3
forces = [
    # forward
    {
        'step': 5,
        'end_step': 6,
        'force': 30.0,
        'times': times,
        'direction': np.array([1,0,0]),
        'success': [False]*times
    },
    {
        'step': 5,
        'end_step': 6,
        'force': 40.0,
        'times': times,
        'direction': np.array([1,0,0]),
        'success': [False]*times
    },
    {
        'step': 5,
        'end_step': 6,
        'force': 50.0,
        'times': times,
        'direction': np.array([1,0,0]),
        'success': [False]*times
    },
    {
        'step': 5,
        'end_step': 6,
        'force': 60.0,
        'times': times,
        'direction': np.array([1,0,0]),
        'success': [False]*times
    },
    {
        'step': 5,
        'end_step': 6,
        'force': 70.0,
        'times': times,
        'direction': np.array([1,0,0]),
        'success': [False]*times
    },
    {
        'step': 5,
        'end_step': 6,
        'force': 85.0,
        'times': times,
        'direction': np.array([1,0,0]),
        'success': [False]*times
    },
    #backward
    {
        'step': 5,
        'end_step': 6,
        'force': 30.0,
        'times': times,
        'direction': np.array([-1,0,0]),
        'success': [False]*times
    },
    {
        'step': 5,
        'end_step': 6,
        'force': 40.0,
        'times': times,
        'direction': np.array([-1,0,0]),
        'success': [False]*times
    },
    {
        'step': 5,
        'end_step': 6,
        'force': 50.0,
        'times': times,
        'direction': np.array([-1,0,0]),
        'success': [False]*times
    },
    {
        'step': 5,
        'end_step': 6,
        'force': 60.0,
        'times': times,
        'direction': np.array([-1,0,0]),
        'success': [False]*times
    },
    {
        'step': 5,
        'end_step': 6,
        'force': 70.0,
        'times': times,
        'direction': np.array([-1,0,0]),
        'success': [False]*times
    },
    {
        'step': 5,
        'end_step': 6,
        'force': 85.0,
        'times': times,
        'direction': np.array([-1,0,0]),
        'success': [False]*times
    },
    #right
    {
        'step': 5,
        'end_step': 6,
        'force': 30.0,
        'times': times,
        'direction': np.array([0,1,0]),
        'success': [False]*times
    },
    {
        'step': 5,
        'end_step': 6,
        'force': 40.0,
        'times': times,
        'direction': np.array([0,1,0]),
        'success': [False]*times
    },
    {
        'step': 5,
        'end_step': 6,
        'force': 50.0,
        'times': times,
        'direction': np.array([0,1,0]),
        'success': [False]*times
    },
    {
        'step': 5,
        'end_step': 6,
        'force': 60.0,
        'times': times,
        'direction': np.array([0,1,0]),
        'success': [False]*times
    },
    {
        'step': 5,
        'end_step': 6,
        'force': 70.0,
        'times': times,
        'direction': np.array([0,1,0]),
        'success': [False]*times
    },
    {
        'step': 5,
        'end_step': 6,
        'force': 85.0,
        'times': times,
        'direction': np.array([0,1,0]),
        'success': [False]*times
    },
    #left
    {
        'step': 5,
        'end_step': 6,
        'force': 30.0,
        'times': times,
        'direction': np.array([0,-1,0]),
        'success': [False]*times
    },
    {
        'step': 5,
        'end_step': 6,
        'force': 40.0,
        'times': times,
        'direction': np.array([0,-1,0]),
        'success': [False]*times
    },
    {
        'step': 5,
        'end_step': 6,
        'force': 50.0,
        'times': times,
        'direction': np.array([0,-1,0]),
        'success': [False]*times
    },
    {
        'step': 5,
        'end_step': 6,
        'force': 60.0,
        'times': times,
        'direction': np.array([0,-1,0]),
        'success': [False]*times
    },
    {
        'step': 5,
        'end_step': 6,
        'force': 70.0,
        'times': times,
        'direction': np.array([0,-1,0]),
        'success': [False]*times
    },
    {
        'step': 5,
        'end_step': 6,
        'force': 85.0,
        'times': times,
        'direction': np.array([0,-1,0]),
        'success': [False]*times
    },
]
