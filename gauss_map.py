import numpy as np
import matplotlib as ml
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def _make_arc(p0, p1):
    '''
    Assumes p0 and p1 lies on a unit sphere and
    then creates a circular arc line between
    p0 and p1.
    '''
    n = np.cross(p0,p1)      # rotation axis from p0 to p1
    s = np.linalg.norm(n)    # sinus theta
    c = np.dot(p0,p1)        # cos theta
    theta = np.arctan2(s,c)  # rotation angle
    N = 32                   # number of segments of the arc
    dtheta = theta/N         # The rotation angle covered by each segment
    X = np.zeros(N+1,dtype=np.float64)
    Y = np.zeros(N+1,dtype=np.float64)
    Z = np.zeros(N+1,dtype=np.float64)
    alpha = 0
    for i in range(N+1):
        radians = i*dtheta
        s = radians/theta
        q = (1-s)*p0 + s*p1
        p = q / np.linalg.norm(q)
        X[i] = p[0]
        Y[i] = p[1]
        Z[i] = p[2]
    return X,Y,Z


def draw(N, S, E):
    '''
    Draws a matplotlib figure illustrating the discrete gauss
    map and its associated tangent cone of a vertex centered
    triangle patch. N is an array of normals of the faces around
    the vertex, E is the outward going  edge directions from the
    vertex, and S is the curvature sign of the edges in E.
    '''
    fig = plt.figure(figsize=(8,6))
    fig.clf()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('Discrete Gauss Map')
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_zlim(-1.0, 1.0)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')    
    ax.plot(N[:,0],N[:,1],N[:,2],'ko')
    K = len(N)
    for i in range(K):
        pi = N[i,:]
        style = '-k'    
        nX = [0.0, pi[0]]
        nY = [0.0, pi[1]]
        nZ = [0.0, pi[2]]
        ax.plot(nX,nY,nZ,style)
    for i in range(K):
        pi = N[i,:]
        pj = N[(i+1)%K,:]
        style = '-g'        
        if S[i]>0:
            style = '-b'
        elif S[i]<0:
            style = '-r'                    
        aX,aY,aZ = _make_arc(pi, pj)
        ax.plot(aX,aY,aZ,style)
    for i in range(K):
        style = '-g'        
        ei = E[i,:]
        ej = E[(i+1)%K,:]
        eX = [0.0, ei[0]]
        eY = [0.0, ei[1]]
        eZ = [0.0, ei[2]]
        ax.plot(eX,eY,eZ,style)
        aX,aY,aZ = _make_arc(ei, ej)
        ax.plot(aX,aY,aZ,style)
    plt.show()
