import numpy as np


class VertexPatch:
    
    def __init__(self,V, T, N, S, E):
        self.V = V  # Vertices of the mesh in model frame, last vertex is the "center"
        self.T = T  # Triangles of the vertex mesh patch, first index is always the center vertex 
        self.N = N  # Normals of the triangles (only around the vertex patch)
        self.S = S  # Sign of curvature of the one-ring neighborhood of edges around the center vertex.
        self.E = E  # The edge vectors going out from the center vertex        


class Private:

    @staticmethod
    def create_edges_array(V, T):
        S = len(T)
        E = np.zeros((S,3), dtype=np.float64)
        for s in range(S):
            i = T[s,0]
            j = T[s,1]
            k = T[s,2]
            pi = V[i,:]
            pj = V[j,:]
            e = pj - pi
            E[s,:] = e / np.linalg.norm(e)
        return E


    @staticmethod
    def create_normals_array(V, T):
        S = len(T)
        N = np.zeros((S,3), dtype=np.float64)
        for s in range(S):
            i = T[s,0]
            j = T[s,1]
            k = T[s,2]
            pi = V[i,:]
            pj = V[j,:]
            pk = V[k,:]
            m = np.cross( pj-pi,pk-pi)
            N[s,:] = m / np.linalg.norm(m)
        return N

    @staticmethod
    def create_curvature_info(V, T, N):
        E = len(T)
        S = np.zeros((E,), dtype=np.float64)
        for e in range(E):
            i = T[e,0]
            j = T[e,1]
            k = T[e,2]
            edge = V[k,:] - V[i,:]
            n_before = N[e,:]
            n_after = N[(e+1)%E,:]
            tst = np.dot(edge,np.cross(n_after,n_before))
            if tst>0:
                S[e] = 1.0
            elif tst<0:
                S[e] = -1.0
        return S

    @staticmethod
    def create_closed_mesh(V, T):
        min_coord = np.min(V, axis=0)
        max_coord = np.max(V, axis=0)
        cntV = len(V)
        cntT = len(T)
        # Create new vertices
        newV = np.array(V, copy=True)
        newV[:, 2] = min_coord[2] - 0.1
        # Create new faces
        newT = np.zeros((cntT*3,3),dtype=np.int)
        for s in range(cntT):
            i = T[s,0]  # center vertex
            j = T[s,1]
            k = T[s,2]
            l = i + cntV
            m = j + cntV
            n = k + cntV
            newT[s*3 + 0,:] = [j, m, n]
            newT[s*3 + 1,:] = [n, k, j]
            newT[s*3 + 2,:] = [n, m, l]
        # Combine old and new into results
        VV = np.concatenate((V, newV), axis=0)
        TT = np.concatenate((T,newT), axis=0)
        return VV, TT


def create_spike_vertex_patch(height, radius):
    N = len(radius)
    V = np.zeros((N+1,3), dtype=np.float64)
    T = np.zeros((N,3), dtype=np.int)
    delta_theta = 2.0*np.pi/N
    for i in range(N):
        theta = i*delta_theta
        x = radius[i]*np.cos(theta)
        y = radius[i]*np.sin(theta)
        z = height
        V[i,:] = (x,y,z)
    V[N,:] = (0.0,0.0,0.0)
    for i in range(N):
        T[i,:] = (N, (i+1)%N, (i+2)%N)
    N = Private.create_normals_array(V,T)
    E = Private.create_edges_array(V,T)
    S = Private.create_curvature_info(V,T,N)
    patch = VertexPatch(V, T, N, S, E)
    VV, TT = Private.create_closed_mesh(V, T)
    return VV, TT, patch


def create_saddle_vertex_patch(height, radius):
    N = len(height)
    V = np.zeros((N+1,3), dtype=np.float64)
    T = np.zeros((N,3), dtype=np.int)
    delta_theta = 2.0*np.pi/N
    for i in range(N):
        theta = i*delta_theta
        x = radius*np.cos(theta)
        y = radius*np.sin(theta)
        z = height[i]
        V[i,:] = (x,y,z)
    V[N,:] = (0.0,0.0, 0.0)
    for i in range(N):
        T[i,:] = (N, (i+1)%N, (i+2)%N)
    N = Private.create_normals_array(V,T)
    E = Private.create_edges_array(V,T)
    S = Private.create_curvature_info(V,T,N)
    patch = VertexPatch(V, T, N, S, E)
    VV, TT = Private.create_closed_mesh(V, T)
    return VV, TT, patch


def create_polygon_vertex_patch(points):
    N = len(points)
    V = np.zeros((N+1,3), dtype=np.float64)
    T = np.zeros((N,3), dtype=np.int)
    for i in range(N):
        V[i,:] = points[i,:]
    V[N,:] = (0.0, 0.0, 0.0)
    for i in range(N):
        T[i,:] = (N, (i+1)%N, (i+2)%N)
    N = Private.create_normals_array(V,T)
    E = Private.create_edges_array(V,T)
    S = Private.create_curvature_info(V,T,N)
    patch = VertexPatch(V, T, N, S, E)
    VV, TT = Private.create_closed_mesh(V, T)
    return VV, TT, patch