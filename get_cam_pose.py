import numpy as np

def compute_v2():
    lookat = np.array([0.4, 0.0, 0.6])
    pos_old = np.array([1.421, -0.000, 1.436])
    
    diff = pos_old - lookat
    # Move 10% further along the floor
    diff[0] *= 1.1
    diff[1] *= 1.1
    
    pos = lookat + diff
    
    fwd = lookat - pos
    fwd /= np.linalg.norm(fwd)
    
    world_up = np.array([0, 0, 1])
    right = np.cross(fwd, world_up)
    right /= np.linalg.norm(right)
    
    up = np.cross(right, fwd)
    
    return pos, fwd, up, right

pos, fwd, up, right = compute_v2()
print("Move 10% further out along the floor")
print("Pos XML: {:.3f} {:.3f} {:.3f}".format(*pos))
print("xyaxes: {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f}".format(*right, *up))
