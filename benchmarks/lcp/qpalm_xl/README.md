# LCP XL QPALM + OSQP
Canonical 25-case panel: LCP7, LCP8, LCP10, LCP11, LCP12 at n=1e6,...,5e6.

QP embedding: min 1/2 x'(L+L')x - g'x subject to x>=0 and Lx>=g. The QP objective equals x'w.
External original-LCP certificate (tol=1e-6): nonnegativity violation <= tol, ||x*w||_inf <= tol, and |x'w| <= tol. Natural residual ||min(x,w)||_inf is also reported. A separate QP KKT1 verdict uses native solver duals.

OSQP is run only on convex LCP7/LCP8/LCP10. LCP11/LCP12 are marked unsupported_nonconvex without modifying Q.
No x/y vectors are stored in JSON, to avoid multi-GB output at 5M variables.
