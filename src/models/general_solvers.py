import numpy as np

def sinkhorn_knopp(a, b, M, reg, init_u=None, init_v=None, numItermax=1000,
    stopThr=1e-9, verbose=False, log=False, **kwargs):


    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)

    if len(a) == 0:
        a = np.ones((M.shape[0],), dtype=np.float64) / M.shape[0]
    if len(b) == 0:
        b = np.ones((M.shape[1],), dtype=np.float64) / M.shape[1]

    # init data
    Nini = len(a)
    Nfin = len(b)

    if len(b.shape) > 1:
        nbb = b.shape[1]
    else:
        nbb = 0

    if log:
        log = {'err': [], 'failed': False}  # Failed will work as an exit code

    # we assume that no distances are null except those of the diagonal of distances
    if (init_u is not None) and (init_v is not None):
        u = init_u
        v = init_v
    else:  # Usual uniform init
        # we assume that no distances are null except those of the diagonal of
        # distances
        if nbb:
            u = np.ones((Nini, nbb)) / Nini
            v = np.ones((Nfin, nbb)) / Nfin
        else:
            u = np.ones(Nini) / Nini
            v = np.ones(Nfin) / Nfin
    uprev = np.zeros(Nini)
    vprev = np.zeros(Nini)
    K = np.exp(-M / reg)
    Kp = (1 / a).reshape(-1, 1) * K
    it = 0
    err = 1
    while (err > stopThr and it < numItermax):
        uprev = u
        vprev = v
        KtransposeU = np.dot(K.T, u)
        v = np.divide(b, KtransposeU)
        u = 1. / np.dot(Kp, v)

        if (np.any(KtransposeU == 0) or
                np.any(np.isnan(u)) or np.any(np.isnan(v)) or
                np.any(np.isinf(u)) or np.any(np.isinf(v))):
            # we have reached the machine precision
            # come back to previous solution and quit loop
            print('Warning: numerical errors at iteration', it)
            u = uprev
            v = vprev
            log['failed'] = True
            break
        if it % 10 == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            if nbb:
                err = np.sum((u - uprev)**2) / np.sum((u)**2) + \
                    np.sum((v - vprev)**2) / np.sum((v)**2)
            else:
                transp = u.reshape(-1, 1) * (K * v)
                err = np.linalg.norm((np.sum(transp, axis=0) - b))**2
            if log:
                log['err'].append(err)

            if verbose:
                if it % 200 == 0:
                    print(
                        '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(it, err))
        it = it + 1
    if log:
        log['u'] = u
        log['v'] = v
        log['it'] = it
    if nbb:  # return only loss
        res = np.zeros((nbb))
        for i in range(nbb):
            res[i] = np.sum(
                u[:, i].reshape((-1, 1)) * K * v[:, i].reshape((1, -1)) * M)
        if log:
            return res, log
        else:
            return res

    else:  # return OT matrix

        if log:
            return u.reshape((-1, 1)) * K * v.reshape((1, -1)), log
        else:
            return u.reshape((-1, 1)) * K * v.reshape((1, -1))