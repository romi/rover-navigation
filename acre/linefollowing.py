import math
import numpy as np
import cv2
from sklearn.decomposition import PCA

from acre.interfaces import *
from acre.debug import *


class LineDetector(ISensor):
    def __init__(self, camera, segmentation, center_x, center_y, pixels_per_meter, invert): 
        super().__init__()
        self.camera = camera
        self.segmentation = segmentation
        self.cx = int(center_x)
        self.cy = int(center_y)
        self.pixels_per_meter = pixels_per_meter
        self.invert = invert

    @property
    def data_type(self):
        return DataType.TRACK_ERROR

    @property
    def size(self):
        return 2

    @property
    def names(self):
        return ['cross-track-error', 'orientation']

    @property
    def std(self):
        # see stats.{py,txt}
        return [2 * 0.025, 2 * 0.05]

    @property
    def var(self):
        return [self.std[0]**2, self.std[1]**2]
    
    def measure(self):
        return self._estimate_track_error(), self.std
    
    def _estimate_track_error(self):
        image = self.camera.grab()
        debug_store_image("camera", image, "png")
        mask = self.segmentation.compute_mask(image)
        if self.invert:
            mask = cv2.bitwise_not(mask)
        debug_store_image("mask1", mask, "png")
        mask = self._erode_dilate(mask, 5, 15)
        debug_store_image("mask2", mask, "png")
        mask = self._get_center_mask(mask)
        debug_store_image("mask3", mask, "png")

        pixels = np.squeeze(np.where(mask == 255)).T  # get coordinates of white pixels
        number_pixels, _ = pixels.shape
        if number_pixels == 0:
            cross_track_error = 0.0
            orientation = 0.0
        else:
            center, pc0, pc1 = self._pca(pixels)
            cross_track_error = self._get_cross_track_error([self.cy, self.cx], center, pc0)
            orientation = self._get_rel_orientation(center, pc0)            
            self._store_pca("camera", image, center, pc0, pc1,
                            cross_track_error, orientation)
            self._store_pca("mask", cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB),
                            center, pc0, pc1, cross_track_error, orientation)

        cross_track_error /= self.pixels_per_meter
        return np.array([cross_track_error, orientation])

    def _erode_dilate(self, mask, er_it=5, dil_it=25):
        kernel = np.array([[0, 1, 0],
                           [1, 1, 1],
                           [0, 1, 0]]).astype(np.uint8)
        
        res = cv2.erode(mask, kernel, iterations=er_it)
        res = cv2.dilate(res, kernel, iterations=dil_it)
        return res

    def _pca(self, pixels):
        pca = PCA(n_components=2)
        pca.fit(pixels)
        center = pca.mean_
        v = pca.explained_variance_
        pcs = pca.components_
        pc0 = center + pcs[0] * np.sqrt(v[0])
        pc1 = center + pcs[1] * np.sqrt(v[1])
        return center, pc0, pc1

    def _get_center_mask(self, mask):
        h, w = mask.shape
        x1 = self.cx - 200
        x2 = self.cx + 200
        y1 = int(0.1 * h)
        y2 = h - int(0.1 * h)
        # Crop central window of (W, H) = (1000, 800)
        center_mask = mask
        center_mask[0:y1, :] = 0
        center_mask[y2:h, :] = 0
        center_mask[:, 0:x1] = 0
        center_mask[:, x2:w] = 0
        return center_mask

    def _get_sign_cte(self, x0, x1, x2):
        a = -(x2[1] - x1[1]) / (x2[0] - x1[0])
        c = -(a * x1[0] + x1[1])
        return np.sign(a * x0[0] + x0[1] + c)

    def _get_cross_track_error(self, x0, x1, x2):
        s = self._get_sign_cte(x0, x1, x2)
        return (-s
                * np.abs((x2[0] - x1[0]) * (x1[1] - x0[1])
                         - (x1[0] - x0[0]) * (x2[1] - x1[1]))
                / np.sqrt((x2[0] - x1[0])**2 + (x2[1] - x1[1])**2))

    def _get_rel_orientation(self, x1, x2):
        pts = np.array([x1, x2])
        pts = pts[pts[:, 0].argsort()]
        return np.arctan2(pts[0][0] - pts[1][0], pts[0][1] - pts[1][1]) + np.pi/2

    def _store_pca(self, label, im, center, pc0, pc1, cte, orientation):
        cv2.line(im, (self.cx, 0), (self.cx, self.cy), (0, 255, 0), 5)
        cv2.line(im, (0, self.cy), (self.cx, self.cy), (0, 255, 0), 5)
        cv2.line(im,
                 tuple(center[::-1].astype(np.int32)),
                 tuple(pc0[::-1].astype(np.int32)),
                 (0, 0, 255), 5)
        cv2.line(im,
                 tuple(center[::-1].astype(np.int32)),
                 tuple(pc1[::-1].astype(np.int32)),
                 (255, 0, 0), 5)
        cv2.putText(im, f"CTE:{cte:0.3f}", (100, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 4, color=(0, 0, 255), thickness=10)
        cv2.putText(im, f"OR.:{orientation*180/np.pi:0.3f}", (100, 1000),
                    cv2.FONT_HERSHEY_SIMPLEX, 4, color=(0, 0, 255), thickness=10)
        debug_store_image(f"{label}-components", im, "png")

        
class SVM(ISegmentation):
    def __init__(self, coefficients, intercept):
        super().__init__()
        self.w = coefficients
        self.intercept = intercept
        
    def compute_mask(self, image):
        val = (self.w[0] * image[:, :, 0]
               + self.w[1] * image[:, :, 1]
               + self.w[2] * image[:, :, 2]
               + self.intercept)
        mask = (val > 0) * 255
        return mask.astype(np.uint8)


def draw_map(map_image, rover, pixels_per_meter, cross_track_error,
             orientation, radius):
    map_height, map_width, _ = map_image.shape
    im = map_image.copy()
    length = rover.dimensions[1]
    # The rover's position (x,y) is measured between the back
    # wheels. The camera is position at the center of the rover
    # at (x+length/2, y)
    xc = (rover.x + length/2.0 * math.cos(rover.orientation)) * pixels_per_meter
    yc = (rover.y + length/2.0 * math.sin(rover.orientation)) * pixels_per_meter
    yc = map_height - yc
    w, h = rover.dimensions
    w *= pixels_per_meter
    h *= pixels_per_meter
    r = RotatedRectangle((xc, yc), (w, h), rover.orientation) 
    r.draw(im)
    s = f"Rover({rover.x:0.3f}, {rover.y:0.3f}, {np.degrees(rover.orientation):0.3f})"
    s += f" -- CTE:{cross_track_error:0.3f}, angle: {orientation:0.3f}"
    if radius is None:
        s += f", R: -"
    else:
        s += f", R: {radius:0.3f}"
    cv2.putText(im, s, (1000, 1000), cv2.FONT_HERSHEY_SIMPLEX, 4,
                color=(0, 0, 0), thickness=10)
    debug_store_image("map", im, "jpg")
        

class RotatedRectangle:
    def __init__(self, center, size, angle):
        (self.W, self.H) = size   # rectangle width and height
        self.d = math.sqrt(self.W**2 + self.H**2)/2.0  # distance from center to vertices
        # self.c = (int(p0[0]+self.W/2.0),int(p0[1]+self.H/2.0)) # center point coordinates
        self.center = center
        self.alpha = angle  # rotation angle in radians
        self.beta = math.atan2(self.H, self.W)  # angle between d and horizontal axis
        # Center Rotated vertices in image frame
        p0 = (int(self.center[0] - self.d * math.cos(self.beta - self.alpha)),
              int(self.center[1] - self.d * math.sin(self.beta - self.alpha))) 
        p1 = (int(self.center[0] - self.d * math.cos(self.beta + self.alpha)),
              int(self.center[1] + self.d * math.sin(self.beta + self.alpha))) 
        p2 = (int(self.center[0] + self.d * math.cos(self.beta - self.alpha)),
              int(self.center[1] + self.d * math.sin(self.beta - self.alpha))) 
        p3 = (int(self.center[0] + self.d * math.cos(self.beta + self.alpha)),
              int(self.center[1] - self.d * math.sin(self.beta + self.alpha))) 
        self.verts = [p0, p1, p2, p3]
        
    def draw(self, image):
        for i in range(len(self.verts)-1):
            cv2.line(image,
                     (self.verts[i][0], self.verts[i][1]),
                     (self.verts[i+1][0], self.verts[i+1][1]),
                     (0, 0, 0), 5)
            cv2.line(image,
                     (self.verts[3][0], self.verts[3][1]),
                     (self.verts[0][0], self.verts[0][1]),
                     (0, 0, 0), 5)
