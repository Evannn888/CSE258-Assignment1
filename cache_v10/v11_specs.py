# Bayesian FM chains (GPU).  Task 1: rank 64 Gaussian, rank 64 ordinal probit, two rank 32 Gaussian (one of them
# 1500 sweeps); Task 3: rank 32 Gaussian with month and session slots, rank 32 and rank 64 ordinal probit with
# month, session and user-year slots.
BFM_SPECS = {1: (dict(k=64, iters=500, burn=100, seed=1), dict(k=64, iters=500, burn=100, probit=True, seed=2),
                 dict(k=32, iters=1500, burn=300, seed=5), dict(k=32, iters=500, burn=100, seed=6)),
             3: (dict(k=32, iters=500, burn=100, time=('month', 'session'), seed=1),
                 dict(k=32, iters=500, burn=100, probit=True, time=('month', 'session', 'uyear'), seed=3),
                 dict(k=64, iters=500, burn=100, probit=True, time=('month', 'session', 'uyear'), seed=4))}
