"""Gravity-aligned camera placement around the statue, in SfM world units.

Angles are defined in the orbit frame: "up" is the mean camera up vector, and
azimuth 0 is the statue's front (the direction to the first inner-ring frame,
which faces her; Low Library is behind her in that view). Azimuth 90 is her
left side and 270 her right side (the scepter is in her right hand), checked
on renders; `describe_azimuth` turns an angle into those words.

All captures were taken at eye level, so views from well above the statue are
outside what the splat was trained on (floaters, smeared ground). `orbit_view`
therefore clamps camera height to the band the capture covered.
"""

from dataclasses import dataclass

import numpy as np

from scene import View, orbit_frame

FRONT_VIEW = "r2_IMG_5820_0001000.jpg"


@dataclass
class OrbitFrame:
    centre: np.ndarray
    up: np.ndarray
    front: np.ndarray   # horizontal unit vector from the statue toward the viewer at azimuth 0
    side: np.ndarray    # up x front
    height_band: tuple  # (min, max) camera height along `up`, absolute

    @classmethod
    def from_views(cls, views, margin=0.3):
        centre, up = orbit_frame(views)
        d = views[FRONT_VIEW].centre - centre
        front = d - (d @ up) * up
        front /= np.linalg.norm(front)
        h = np.array([v.centre @ up for v in views.values()])
        return cls(centre, up, front, np.cross(up, front), (h.min() - margin, h.max() + margin))

    def direction(self, azimuth_deg, elevation_deg):
        az, el = np.radians(azimuth_deg), np.radians(elevation_deg)
        horiz = np.cos(az) * self.front + np.sin(az) * self.side
        return np.cos(el) * horiz + np.sin(el) * self.up

    def angles_of(self, point, target):
        """(azimuth, elevation) in degrees of `point` as seen from `target`."""
        d = point - target
        h = d - (d @ self.up) * self.up
        az = np.degrees(np.arctan2(h @ self.side, h @ self.front)) % 360
        el = np.degrees(np.arctan2(d @ self.up, np.linalg.norm(h)))
        return az, el


def look_at(position, target, up, width=1280, height=720, hfov_deg=50.0):
    """Pinhole View at `position` looking at `target`, horizon level w.r.t. `up`."""
    z = target - position
    z /= np.linalg.norm(z)
    x = np.cross(-up, z)          # OpenCV axes: x right, y down, z forward
    if np.linalg.norm(x) < 1e-6:  # looking straight up/down
        x = np.cross(np.array([1.0, 0, 0]), z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    f = 0.5 * width / np.tan(np.radians(hfov_deg) / 2)
    K = np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]])
    return View("novel", R, -R @ position, K, width, height)


def orbit_view(frame, target, distance, azimuth_deg, elevation_deg, clamp_height=True, **kw):
    position = target + distance * frame.direction(azimuth_deg, elevation_deg)
    if clamp_height:
        h = position @ frame.up
        position = position + (np.clip(h, *frame.height_band) - h) * frame.up
    return look_at(position, target, frame.up, **kw)


def describe_azimuth(az):
    """Plain-language side of the statue for a viewer standing at this azimuth."""
    names = ["front", "front, her left", "her left side", "back, her left",
             "back", "back, her right", "her right side", "front, her right"]
    return names[int(((az % 360) + 22.5) // 45) % 8]
