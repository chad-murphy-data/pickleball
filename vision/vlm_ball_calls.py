"""Ball-position calls on four localization grids (2026-08-19).

Recorded by eye from the 3x3 grids, as (x_frac, y_frac) WITHIN a cell
so they survive any display scaling; None = "I can't see it". Rendered
for the user to check with scratch ball_mark.py. Unverified at time of
writing: a WRONG circle is worse news than a missing one, so these are
claims awaiting audit, not a measurement yet.
"""
BALL = {
 1: {1:(0.573,0.244), 2:(0.508,0.264), 3:(0.433,0.286), 4:(0.363,0.361),
     5:(0.298,0.422), 6:(0.271,0.486), 7:(0.229,0.367), 8:None,
     9:(0.261,0.328)},
13: {1:(0.443,0.361), 2:None,          3:None,          4:(0.401,0.181),
     5:(0.422,0.181), 6:None,          7:None,          8:(0.370,0.208),
     9:None},
24: {1:(0.401,0.089), 2:None,          3:None,          4:(0.258,0.486),
     5:(0.206,0.583), 6:(0.223,0.444), 7:(0.290,0.278), 8:None,
     9:(0.385,0.125)},
27: {1:(0.548,0.436), 2:(0.599,0.561), 3:(0.718,0.694), 4:(0.706,0.583),
     5:(0.687,0.411), 6:(0.679,0.292), 7:(0.658,0.208), 8:(0.613,0.236),
     9:(0.672,0.194)},
}
