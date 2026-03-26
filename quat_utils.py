import numpy as np

def quatmultiply(q1, q2):
    """
    Multiply two quaternions.
    Input format: [w, x, y, z]
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    
    return np.array([w, x, y, z])

# Function for quaternion conjugate (equivalent to MATLAB's quatconj)
def quatconj(q):
    """
    Compute quaternion conjugate.
    Input format: [w, x, y, z]
    """
    return np.array([q[0], -q[1], -q[2], -q[3]])