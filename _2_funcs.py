import cv2
import numpy as np
import _3_gvars as gvars

# GUI funcs---------------------------------------------------------------------------------------------------
def selectRoi(idSelected):
    gvars.selectedRoiId = idSelected
    cv2.setTrackbarPos('minDepth', 'Roi', gvars.l_rois[idSelected].t_minMaxDepth[0])
    cv2.setTrackbarPos('maxDepth', 'Roi', gvars.l_rois[idSelected].t_minMaxDepth[1])
    cv2.setTrackbarPos('brightestSizeX', 'Roi', int(gvars.l_rois[idSelected].t_brightestRegionSize[0]))
    cv2.setTrackbarPos('brightestSizeY', 'Roi', int(gvars.l_rois[idSelected].t_brightestRegionSize[1]))
    cv2.setTrackbarPos('numBrightestRegions', 'Roi', int(gvars.l_rois[idSelected].numBrightestRegions))
    cv2.setTrackbarPos('maxDistanceBetweenPoints', 'Roi', int(gvars.l_rois[idSelected].maxDistanceBetweenPoints))

    val = int(gvars.l_rois[idSelected].maxValFilterFactor * 100)
    cv2.setTrackbarPos('maxValFilterFactor', 'Roi', val)
    val = int(gvars.l_rois[idSelected].overRegionsFactor * 100)
    cv2.setTrackbarPos('overRegionsFactor', 'Roi', val)

def update_maxValFilterFactor(val):
    fVal = float(val) / 100.0 
    gvars.l_rois[gvars.selectedRoiId].maxValFilterFactor = fVal

def update_minDepth(val):
    newTuple = (val, + gvars.l_rois[gvars.selectedRoiId].t_minMaxDepth[1])
    gvars.l_rois[gvars.selectedRoiId].t_minMaxDepth = newTuple

def update_maxDepth(val):
    newTuple = (gvars.l_rois[gvars.selectedRoiId].t_minMaxDepth[0], val)
    gvars.l_rois[gvars.selectedRoiId].t_minMaxDepth = newTuple
    
def update_brightestRegionX(val):
    newTuple = (val, + gvars.l_rois[gvars.selectedRoiId].t_brightestRegionSize[1])
    gvars.l_rois[gvars.selectedRoiId].t_brightestRegionSize = newTuple

def update_brightestRegionY(val):
    newTuple = (gvars.l_rois[gvars.selectedRoiId].t_brightestRegionSize[0], val)
    gvars.l_rois[gvars.selectedRoiId].t_brightestRegionSize = newTuple
    
def update_numBrightestRegions(val):
    gvars.l_rois[gvars.selectedRoiId].numBrightestRegions = val

def update_overRegionsFactor(val):
    fVal = float(val) / 100.0 
    gvars.l_rois[gvars.selectedRoiId].overRegionsFactor = fVal

def update_maxDistanceBetweenPoints(val):
    gvars.l_rois[gvars.selectedRoiId].maxDistanceBetweenPoints = val

# GET funcs---------------------------------------------------------------------------------------------------
def getAllFilteredRoi(id, frame, filterPerMaxVal = True, withinRegion = None):
    roidFrame = frame.copy()
    filtered_image = frame.copy()

    if gvars.l_rois:
        depthFilteredFrame = gvars.l_rois[id].getDepthFilteredFrame(frame)
        roidFrame = gvars.l_rois[id].getRoiMaskedFrame(depthFilteredFrame)

        if withinRegion:
            mask = np.zeros_like(roidFrame, dtype=np.uint8)
            x1, y1, x2, y2 = withinRegion

            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1) 

            roidFrame = cv2.bitwise_and(roidFrame, roidFrame, mask=mask)

        if filterPerMaxVal is True:
            max_depth_value = np.max(roidFrame)
            threshold_value = max_depth_value - (max_depth_value/gvars.l_rois[id].maxValFilterFactor)

            # Threshold the depth image
            _, mask = cv2.threshold(roidFrame, threshold_value, 255, cv2.THRESH_BINARY)

            # Apply the mask to the original filtered image
            filtered_image = cv2.bitwise_and(depthFilteredFrame, depthFilteredFrame, mask=mask)
        else:
            filtered_image = roidFrame

    return filtered_image

def findClosestPoint(regions, targetCentroidsAbs, maxDistance = 50):
    if not regions:
        raise ValueError("Regions must be non-empty lists of the same length.")
    
    centroidsAbsolute = np.array([
        (region.centroid[0] + region.l_points[0], region.centroid[1] + region.l_points[1]) 
        for region in regions
        if (region.centroid is not None)
    ])
    
    # Compute the pairwise distances between all target centroids and all main centroids
    distances = np.linalg.norm(centroidsAbsolute[:, np.newaxis, :] - targetCentroidsAbs[np.newaxis, :, :], axis=2)
    
    # Find the index of the minimum distance
    minIdx = np.unravel_index(np.argmin(distances), distances.shape)

    # Get the minimum distance
    minDistance = distances[minIdx]
    
    # Get the closest centroid and region
    closestRegionIndex = minIdx[0]
    closestRegion = regions[closestRegionIndex]

    noCloseRegionsFound = False

    if maxDistance is not None and minDistance >= maxDistance:
        noCloseRegionsFound = True
    
    return closestRegion, closestRegionIndex, noCloseRegionsFound

def predictNextCentroid(l_lastCentroids):
        if len(l_lastCentroids) < 2:
            return None  # Not enough data to make a prediction

        # Prepare data for linear regression
        x_data = np.arange(len(l_lastCentroids))
        y_data_x = np.array([centroid[0] for centroid in l_lastCentroids])
        y_data_y = np.array([centroid[1] for centroid in l_lastCentroids])

        # Perform linear regression for x and y coordinates separately
        coef_x = np.polyfit(x_data, y_data_x, 1)
        coef_y = np.polyfit(x_data, y_data_y, 1)

        # Predict the next centroid position
        next_x = np.polyval(coef_x, len(l_lastCentroids))
        next_y = np.polyval(coef_y, len(l_lastCentroids))

        return (int(next_x), int(next_y))

# DRAW funcs---------------------------------------------------------------------------------------------------
def drawRoiPoints(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        gvars.l_rois[gvars.selectedRoiId].l_points.append((x, y))
    
    elif event == cv2.EVENT_RBUTTONDOWN:
        if len(gvars.l_rois[gvars.selectedRoiId].l_points) > 0:
            gvars.l_rois[gvars.selectedRoiId].l_points.pop()

def drawRoisOutlines(frameD, selectedRoi):
    for roi in gvars.l_rois:
        flag = True
        for point in roi.l_points:
            cv2.circle(frameD, point, 3, (200, 200, 200), 4)
            if flag:
                flag = False
                color = (200, 200, 200)
                if roi.id == selectedRoi:
                    color = gvars.roiSelectedColor
                cv2.putText(frameD, str(roi.id), point, cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 1, cv2.LINE_AA)
        if len(roi.l_points) > 1:
            color = (120, 120, 120)
            if roi.id == selectedRoi:
                color = gvars.roiSelectedColor
            cv2.polylines(frameD, [np.array(roi.l_points)], False, color, thickness=2)

def drawAlreadyFoundRegionsAndCentroids(frameD, regions, color = (180, 0, 180), thickness = 1):
    i = 0
    for region in regions:
        x1, y1, x2, y2 = region.l_points

        region.draw(frameD, color, thickness, color, thickness)

        cv2.putText(frameD, str(i), (x1 - 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, color, thickness, cv2.LINE_AA)

        i += 1

def drawBrightestRegions(frameD, l_regionsToDraw):
    i = 0
    colorRect = (0, 255, 0) # brightest
    sizeRect = 2 # brightest
    colorCentroid = (0, 255, 0) # brightest
    sizeCentroid = 3

    if len(l_regionsToDraw) > 0:
        for region in l_regionsToDraw:
            i += 1

            #frameD, colorReg, thicknessReg, colorCentroid, sizeCentroid
            region.draw(frameD, colorRect, sizeRect, colorCentroid, sizeCentroid)

            if i == 1:
                colorRect = 255
                sizeRect = 0
                colorCentroid = (200, 200, 200)
                sizeCentroid = 0

# ---- funcoes inutilizadas por hora

def getOverlapBetweenTwoRegions(region1, region2):
    x1_min, y1_min, x1_max, y1_max = region1
    x2_min, y2_min, x2_max, y2_max = region2

    x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
    y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))

    return x_overlap * y_overlap

def isRegionClustered(regionsToSearch, mainRegionId, overlapThreshold = 0, minRegionsForCluster = 2):
    if overlapThreshold == 0:
        overlapThreshold = gvars.l_rois[mainRegionId].overRegionsFactor - (gvars.l_rois[mainRegionId].overRegionsFactor / 10)
        
    mainRegionArea = (regionsToSearch[mainRegionId][2] - regionsToSearch[mainRegionId][0]) * (regionsToSearch[mainRegionId][3] - regionsToSearch[mainRegionId][1])
    overlapCount = 0

    curRegionId = 0
    l_overlappingRegionsIds = []

    for region in regionsToSearch:
        if curRegionId == mainRegionId:
            curRegionId += 1
            continue

        overlapArea = getOverlapBetweenTwoRegions(region, regionsToSearch[mainRegionId])
        overlapPercent = overlapArea / mainRegionArea

        print(overlapPercent, overlapThreshold)

        if overlapPercent >= overlapThreshold:
            overlapCount += 1
            l_overlappingRegionsIds.append(curRegionId)
        
        curRegionId += 1

    if overlapCount >= minRegionsForCluster:
        return True, l_overlappingRegionsIds
    else:
        return False, []
    
def getClusterRegion(clusteredRegions):
    if not clusteredRegions:
        return None

    x_min = min(region[0] for region in clusteredRegions)
    y_min = min(region[1] for region in clusteredRegions)
    x_max = max(region[2] for region in clusteredRegions)
    y_max = max(region[3] for region in clusteredRegions)

    return (x_min, y_min, x_max, y_max)