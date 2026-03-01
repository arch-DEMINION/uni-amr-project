import numpy as np

times = 1
gravity = [
    {
        'times': times,
        'value': np.array([i*10/100, j*10/100]),
        'success': [False]*times,
        'stop_step': [0,0,0]
    } for i, j in zip([0.06, -0.06, 0, 0]*6*4, [0, 0, 0.06, -0.06]*6*4)
]