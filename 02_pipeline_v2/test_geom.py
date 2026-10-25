import numpy as np, geom
# 1) round trip on a grid of ground points, several pitches/heights
for h in (2.2, 2.5, 2.9):
    for pitch in (0.0, -10.0):
        R, F = np.meshgrid(np.linspace(-15, 15, 13), np.linspace(2, 30, 15))
        u, v = geom.ground_to_pixels(R, F, h, pitch)
        R2, F2, ok = geom.pixels_to_ground(u, v, h, pitch)
        m = ok & (u >= 0) & (u < 640) & (v >= 0) & (v < 640)
        assert m.sum() > 40 and np.allclose(R[m], R2[m], atol=1e-6) and np.allclose(F[m], F2[m], atol=1e-6), (h, pitch)
# 2) a known geometry: horizon row for pitch 0 is v=320; ground 10 m ahead, h=2.5, pitch 0 -> v = 320 + f*h/F
f = geom.focal_px()
u, v = geom.ground_to_pixels(np.array([0.0]), np.array([10.0]), 2.5, 0.0)
assert abs(v[0] - (320 + f * 2.5 / 10.0)) < 1e-9 and abs(u[0] - 320) < 1e-9
# 3) compass rotation: camera facing East (90 deg): forward 10 m -> +10 E ; right 3 m -> 3 m South (-N)
E, N = geom.cam_to_world(np.array([3.0]), np.array([10.0]), 90.0, 0.0, 0.0)
assert np.allclose([E[0], N[0]], [10.0, -3.0], atol=1e-9)
# 4) axis frame: axis pointing north; a point 5 m to the west is 5 m LEFT
s, l = geom.world_to_axis(np.array([-5.0]), np.array([2.0]), 0, 0, 0.0)
assert np.allclose([s[0], l[0]], [2.0, 5.0])
# 5) pixel scale sanity: a 1.8 m wide object at F=8 m covers  f*1.8/8 px laterally (pitch 0)
u1, _ = geom.ground_to_pixels(np.array([-0.9]), np.array([8.0]), 2.5, 0.0); u2, _ = geom.ground_to_pixels(np.array([0.9]), np.array([8.0]), 2.5, 0.0)
assert abs((u2 - u1)[0] - f * 1.8 / 8) < 1e-9
print("geom tests passed; f_px =", round(f, 1))
