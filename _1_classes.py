import numpy as np
import cv2
import _3_gvars as gvars
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

        self.lastRegionFoundAsItem = None
        self.lastRegionFoundAsItemId = -1

        self.overlapJumpToNextRoi = False
        self.overlappingAreaWithNextRoiMask = None

        self.l_directedRegionToSearch = []

        self.lookingForClusters = False
        self.l_clusterRegion = []

    def changeRois(self, l_newRois):
        self.l_orderedRois = l_newRois

    def allowOverlapRoiChange(self):
        self.overlapJumpToNextRoi = True

        currentRoiId = self.l_orderedRois[self.curRoi]
        nextRoiId = self.l_orderedRois[self.curRoi + 1]

        currentRoiPolygon = gvars.l_rois[currentRoiId].getLPointsAsPolygon()
        nextRoiPolygon = gvars.l_rois[nextRoiId].getLPointsAsPolygon()

        overlap_mask, overlap_area = self.calculate_polygon_overlap(currentRoiPolygon, nextRoiPolygon, gvars.frameShape)

        if overlap_area <= 0:
            raise ValueError("No overlapping area between Rois.")
        else:
            self.overlappingAreaWithNextRoiMask = overlap_mask

    def calculate_polygon_overlap(self, polygon1, polygon2, image_shape):
        if not polygon1.size or not polygon2.size:
            raise ValueError("One or both polygons are empty.")
        
        # Create masks for the two polygons
        mask1 = np.zeros(image_shape, dtype=np.uint8)
        mask2 = np.zeros(image_shape, dtype=np.uint8)
        
        cv2.fillPoly(mask1, [polygon1], 255)
        cv2.fillPoly(mask2, [polygon2], 255)
        
        # Calculate overlap
        overlap = cv2.bitwise_and(mask1, mask2)
        overlap_area = np.sum(overlap > 0)
        
        return overlap, overlap_area
    
    def checkItemInNextRoiOverlap(self, overlapPercentMin = 35):
        x1, y1, x2, y2 = self.lastRegionFoundAsItem.l_points

        # Ensure coordinates are within bounds
        if x1 < 0 or y1 < 0 or x2 > self.overlappingAreaWithNextRoiMask.shape[1] or y2 > self.overlappingAreaWithNextRoiMask.shape[0]:
            print("Region coordinates are out of bounds of the overlap mask.")

        # Extract the part of the overlap mask that corresponds to the region
        region_mask = self.overlappingAreaWithNextRoiMask[y1:y2, x1:x2]

        # Calculate the area of the region
        region_area = (x2 - x1) * (y2 - y1)

        # Calculate the overlap area (non-zero pixels in the region mask)
        overlap_area = np.sum(region_mask > 0)

        # Calculate the overlap percentage
        overlap_percentage = (overlap_area / region_area) * 100

        if(overlap_percentage >= overlapPercentMin):
            self.activateNextRoi()

    def activateNextRoi(self):
        self.curRoi = self.curRoi + 1
        self.overlapJumpToNextRoi = False
        self.overlappingAreaWithNextRoiMask = None
    
    def getImageFrameWithRoiOverlap(self, image, overlap_mask):
        if overlap_mask.size == 0:
            raise ValueError("Overlap mask is empty.")
        
        if image.shape[:2] != overlap_mask.shape:
            # Resize the overlap mask to match the image dimensions if necessary
            overlap_mask = cv2.resize(overlap_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # Convert mask to 3-channel image
        overlap_colored = cv2.cvtColor(overlap_mask, cv2.COLOR_GRAY2BGR)
        
        # Define the color for the overlap (e.g., red)
        color = (0, 0, 255)  # Red color in BGR
        overlap_colored[np.where(overlap_mask > 0)] = color  # Apply the color only where the mask is non-zero
        
        # Overlay the overlap mask on the original image
        image_with_overlap = cv2.addWeighted(image, 1.0, overlap_colored, 0.5, 0)
        
        return image_with_overlap


class Region:
    def __init__(self, points, centroid = None):
        self.l_points = points
        self.centroid = centroid
        self.selectedAsItem = False

    def computeCentroid(self, frameAnalysis):
        x1, y1, x2, y2 = self.l_points

        region = frameAnalysis[y1:y2, x1:x2]
        
        moments = cv2.moments(region)
        if moments["m00"] != 0:
            cX = int(moments["m10"] / moments["m00"])
            cY = int(moments["m01"] / moments["m00"])

            self.centroid = (cX, cY)
        else:
            self.centroid = None
        
        return self.centroid
            
    def draw(self, frameD, colorReg = (180, 0, 180), thicknessReg = 1, colorCentroid = (180, 0, 180), sizeCentroid = 1):
        x1, y1, x2, y2 = self.l_points

        cv2.rectangle(frameD, (x1, y1), (x2, y2), colorReg, thicknessReg)

        if self.centroid is not None:
            cv2.circle(frameD, (self.centroid[0] + x1, self.centroid[1] + y1), sizeCentroid, colorCentroid, 1)
    
    def getCentroidAbsolutePosition(self):
        return (self.centroid[0] + self.l_points[0], self.centroid[1] + self.l_points[1])

class Roi:
    def __init__(self, id):
        self.id = id
        self.l_points = []
        self.t_minMaxDepth = (0, 255)
        self.maxValFilterFactor = 3
        self.maxDistanceBetweenPoints = 13
        self.t_brightestRegionSize = (30, 30)
        self.numBrightestRegions = 1
        self.overRegionsFactor = 0.25
        
        self.l_brightestRegionsFound = []

    def eraseAllPoints(self):
        self.l_points = []

    def eraseLastPoint(self):
        if self.l_points:
            self.l_points.pop()

    def getLPointsAsPolygon(self):
        return np.array(self.l_points, np.int32)
    
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
    
    def getRegionsCentroid(self, frameA, l_regions):
        l_centroids = []

        if len(l_regions) > 0:
            for region in l_regions:
                centroid = region.computeCentroid(frameA)
                
                l_centroids.append(centroid)
        
        return l_centroids
    
    def findBrightestRegions(self, frame, num_regions, overlap_threshold, hasAlreadyFoundRegionsThisFrame = False):
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
        l_brightestRegions = []

        # Create grid of coordinates for the top-left corners of each block
        y_coords, x_coords = np.meshgrid(np.arange(mean_brightness.shape[0]), np.arange(mean_brightness.shape[1]), indexing='ij')

        for idx in flat_indices:
            if len(l_brightestRegions) >= num_regions:
                break

            y, x = np.unravel_index(idx, mean_brightness.shape)
            if not mask[y, x]:
                continue

            # Calculate the coordinates of the region in the original frame
            region = (x_min + x, y_min + y, x_min + x + Nx, y_min + y + Ny)
            l_brightestRegions.append(region)

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

        l_centroids = []
        numRegionsAlreadyFound = 0
        if hasAlreadyFoundRegionsThisFrame:
            numRegionsAlreadyFound = len(self.l_brightestRegionsFound)
        else:
            self.l_brightestRegionsFound = []

        curRegionIndex = 0
        for brightestRegion in l_brightestRegions:
            if (hasAlreadyFoundRegionsThisFrame is not True) | ((hasAlreadyFoundRegionsThisFrame) & (curRegionIndex >= numRegionsAlreadyFound)):
                newRegion = Region(brightestRegion)
                regionCentroid = newRegion.computeCentroid(frame)
                self.l_brightestRegionsFound.append(newRegion)

                l_centroids.append(regionCentroid)

            curRegionIndex += 1

        return l_brightestRegions, l_centroids