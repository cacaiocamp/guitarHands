import numpy as np
import cv2
from skimage.util.shape import view_as_windows

class ItemToTrack:
    def __init__(self, type, roisToTrack):
        self.type = type
        self.isProjecting = False
        self.isActive = False
        self.isBrightestForSure = False
        self.lastRegionAsBasis = False
        self.l_orderedRois = roisToTrack
        self.curRoi = 0

        self.isChangingRoi = False
        self.timeChangingRoi = 0

        self.timeNotFindingCloseRegion = 0
        self.l_lastCentroidsAbs = []
        self.l_predictedCentroidsAbs = []

        self.l_lastRegionFound = []
        self.lastRegionCentroid = (-1, -1)

        self.l_directedRegionToSearch = []

        self.lookingForClusters = False
        self.l_clusterRegion = []

    def changeRois(self, l_newRois):
        self.l_orderedRois = l_newRois

class Region:
    def __init__(self, points, centroid):
        self.l_points = points
        self.centroid = centroid

class Roi:
    def __init__(self, id):
        self.id = id
        self.l_points = []
        self.t_minMaxDepth = (0, 255)
        self.maxValFilterFactor = 3
        self.t_brightestRegionSize = (30, 30)
        self.numBrightestRegions = 1
        self.overRegionsFactor = 0.25
        self.minDistBetweenPoints = 13

    def eraseAllPoints(self):
        self.l_points = []

    def eraseLastPoint(self):
        if self.l_points:
            self.l_points.pop()
    
    def getRoiMaskedFrame(self, frame):
        normalized_masked_frame = frame

        if len(self.l_points) > 1:
            mask = np.zeros_like(frame, dtype=np.uint8)
            cv2.fillPoly(mask, [np.array(self.l_points)], 255)
            
            # Apply the mask
            masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
            
            # Normalize the masked frame for display purposes
            normalized_masked_frame = cv2.normalize(masked_frame, None, 0, 255, cv2.NORM_MINMAX)
            normalized_masked_frame = np.uint8(normalized_masked_frame)

        return normalized_masked_frame
    
    def getDepthFilteredFrame(self, frame):
        filtered_frame = frame
        # Ensure the frame is in a single-channel format
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
        # Create a mask that is True where the depth is within the range
        mask = (frame >= self.t_minMaxDepth[0]) & (frame <= self.t_minMaxDepth[1])
    
        # Normalize the frame to be between 15 and 230
        normalized_frame = np.clip(15 + ((frame - self.t_minMaxDepth[0]) / (self.t_minMaxDepth[1] - self.t_minMaxDepth[0])) * (230 - 15), 15, 230)
        inverted_normalized_frame = 255 - normalized_frame  # Invert the normalized values
    
        # Apply the mask to the normalized frame
        filtered_frame = np.where(mask, inverted_normalized_frame, 0).astype(np.uint8)
    
        return filtered_frame

    def findBrightestRegion(self, frame):
        if self.l_points:
            x_values = [point[0] for point in self.l_points]
            y_values = [point[1] for point in self.l_points]
            x_min, x_max = min(x_values), max(x_values)
            y_min, y_max = min(y_values), max(y_values)

            max_average_brightness = 0
            brightest_region = None

            Nx = self.t_brightestRegionSize[0]
            Ny = self.t_brightestRegionSize[1]

            # Iterate over each NxN pixel block within the ROI
            for y in range(y_min, y_max - Ny, Ny):
                for x in range(x_min, x_max - Nx, Nx):
                    # Get the NxN block
                    block = frame[y:y+Ny, x:x+Nx]
                    
                    # Calculate the average brightness of the block
                    average_brightness = np.mean(block)
                    
                    # Check if this block is brighter than the previous brightest region
                    if average_brightness > max_average_brightness:
                        max_average_brightness = average_brightness
                        brightest_region = (x, y, x+Nx, y+Ny)

            return brightest_region
        else:
            return None
        
    def findBrightestRegions(self, frame, num_regions=2, overlap_threshold=0.25):
        if (not self.l_points) | (len(self.l_points) < 3):
            return []

        x_values = [point[0] for point in self.l_points]
        y_values = [point[1] for point in self.l_points]
        x_min, x_max = min(x_values), max(x_values)
        y_min, y_max = min(y_values), max(y_values)

        Nx = self.t_brightestRegionSize[0]
        Ny = self.t_brightestRegionSize[1]

        # Extract the ROI from the frame
        roi = frame[y_min:y_max, x_min:x_max]

        # Use sliding window approach to get all Nx x Ny blocks
        windows = view_as_windows(roi, (Ny, Nx))

        # Calculate the mean brightness of each block
        mean_brightness = np.mean(windows, axis=(2, 3))

        # Flatten the brightness and get the indices of all regions
        flat_brightness = mean_brightness.flatten()
        flat_indices = np.argsort(-flat_brightness)

        # Create a mask to keep track of valid regions
        mask = np.ones(mean_brightness.shape, dtype=bool)
        brightest_regions = []

        # Create grid of coordinates for the top-left corners of each block
        y_coords, x_coords = np.meshgrid(np.arange(mean_brightness.shape[0]), np.arange(mean_brightness.shape[1]), indexing='ij')

        for idx in flat_indices:
            if len(brightest_regions) >= num_regions:
                break

            y, x = np.unravel_index(idx, mean_brightness.shape)
            if not mask[y, x]:
                continue

            # Calculate the coordinates of the region in the original frame
            region = (x_min + x, y_min + y, x_min + x + Nx, y_min + y + Ny)
            brightest_regions.append(region)

            # Calculate the bounding box of the area to invalidate
            x_start = max(0, x - Nx + 1)
            y_start = max(0, y - Ny + 1)
            x_end = min(mean_brightness.shape[1], x + Nx)
            y_end = min(mean_brightness.shape[0], y + Ny)

            # Create a boolean array of the regions to be invalidated
            invalid_x_coords = x_coords[y_start:y_end, x_start:x_end]
            invalid_y_coords = y_coords[y_start:y_end, x_start:x_end]

            # Calculate overlap areas in a vectorized way
            x1_min, y1_min, x1_max, y1_max = region
            x2_min = x_min + invalid_x_coords
            y2_min = y_min + invalid_y_coords
            x2_max = x2_min + Nx
            y2_max = y2_min + Ny

            x_overlap = np.maximum(0, np.minimum(x1_max, x2_max) - np.maximum(x1_min, x2_min))
            y_overlap = np.maximum(0, np.minimum(y1_max, y2_max) - np.maximum(y1_min, y2_min))
            overlap_area = x_overlap * y_overlap

            overlap_percent = overlap_area / (Nx * Ny)

            # Invalidate regions that have significant overlap
            mask[y_start:y_end, x_start:x_end] &= (overlap_percent < overlap_threshold)

        return brightest_regions