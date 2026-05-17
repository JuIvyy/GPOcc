



import torch



def inverse_sigmoid(x):
    return np.log(x/((1-x)+1e-10))




def _markley_group_mean(qA, qB_nei, w_nei=None):
    """
    qA:        [4]          (this A's quaternion)
    qB_nei:    [K,4]        (neighbors' quaternions for this A)
    w_nei:     [K] or None  (weights per neighbor; if None -> ones)
    returns:   q_avg [4], unit quaternion; sign aligned with qA
    """
    if qB_nei.numel() == 0:
        # no neighbors -> fallback to qA
        qs = qA / qA.norm()
        return qs
    if w_nei is None:
        w_nei = torch.ones(qB_nei.size(0), device=qB_nei.device, dtype=qB_nei.dtype)
    # build 4x4 symmetric M = wA*qA qA^T + sum_k w_k*q_k q_k^T
    # A's own weight (can be a knob; keep 1.0 by default)
    wA = 1.0
    qA = qA / (qA.norm() + 1e-12)
    M = wA * torch.ger(qA, qA)  # [4,4]
    # neighbors
    # out_k = w_k * q_k q_k^T -> scatter-sum (here we just sum in a loop for clarity; K is small per A)
    # If you prefer vectorized: use einsum('ki,kj,k->kij', qB_nei, qB_nei, w_nei).sum(0)
    Mb = torch.einsum('ki,kj,k->ij', qB_nei, qB_nei, w_nei)  # [4,4]
    M = M + Mb
    # principal eigenvector of M
    evals, evecs = torch.linalg.eigh(M)   # ascending
    q = evecs[:, -1]
    # align sign to qA to avoid jumps
    if (q * qA).sum() < 0:
        q = -q
    q = q / (q.norm() + 1e-12)
    return q