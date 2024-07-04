import cv2
import numpy as np
import _3_gvars as gvars

# GUI funcs---------------------------------------------------------------------------------------------------
def selectRoi(idSelected):
    gvars.g_selectedRoi = idSelected
    cv2.setTrackbarPos('minDepth', 'Roi', gvars.glist_rois[idSelected].t_minMaxDepth[0])
    cv2.setTrackbarPos('maxDepth', 'Roi', gvars.glist_rois[idSelected].t_minMaxDepth[1])
    cv2.setTrackbarPos('brightestSizeX', 'Roi', int(gvars.glist_rois[idSelected].t_brightestRegionSize[0]))
    cv2.setTrackbarPos('brightestSizeY', 'Roi', int(gvars.glist_rois[idSelected].t_brightestRegionSize[1]))
    cv2.setTrackbarPos('numBrightestRegions', 'Roi', int(gvars.glist_rois[idSelected].numBrightestRegions))
    cv2.setTrackbarPos('minDistBetweenPoints', 'Roi', int(gvars.glist_rois[idSelected].minDistBetweenPoints))

    val = int(gvars.glist_rois[idSelected].maxValFilterFactor * 100)
    cv2.setTrackbarPos('maxValFilterFactor', 'Roi', val)
    val = int(gvars.glist_rois[idSelected].overRegionsFactor * 100)
    cv2.setTrackbarPos('overRegionsFactor', 'Roi', val)

def update_maxValFilterFactor(val):
    fVal = float(val) / 100.0 
    gvars.glist_rois[gvars.g_selectedRoi].maxValFilterFactor = fVal

def update_minDepth(val):
    newTuple = (val, + gvars.glist_rois[gvars.g_selectedRoi].t_minMaxDepth[1])
    gvars.glist_rois[gvars.g_selectedRoi].t_minMaxDepth = newTuple

def update_maxDepth(val):
    newTuple = (gvars.glist_rois[gvars.g_selectedRoi].t_minMaxDepth[0], val)
    gvars.glist_rois[gvars.g_selectedRoi].t_minMaxDepth = newTuple
    
def update_brightestRegionX(val):
    newTuple = (val, + gvars.glist_rois[gvars.g_selectedRoi].t_brightestRegionSize[1])
    gvars.glist_rois[gvars.g_selectedRoi].t_brightestRegionSize = newTuple

def update_brightestRegionY(val):
    newTuple = (gvars.glist_rois[gvars.g_selectedRoi].t_brightestRegionSize[0], val)
    gvars.glist_rois[gvars.g_selectedRoi].t_brightestRegionSize = newTuple
    
def update_numBrightestRegions(val):
    gvars.glist_rois[gvars.g_selectedRoi].numBrightestRegions = val

def update_overRegionsFactor(val):
    fVal = float(val) / 100.0 
    gvars.glist_rois[gvars.g_selectedRoi].overRegionsFactor = fVal

def update_minDistBetweenPoints(val):
    gvars.glist_rois[gvars.g_selectedRoi].minDistBetweenPoints = val

# GET funcs---------------------------------------------------------------------------------------------------

def computeCentroid(image):
    moments = cv2.moments(image)
    if moments["m00"] != 0:
        cX = int(moments["m10"] / moments["m00"])
        cY = int(moments["m01"] / moments["m00"])

        centroidPos = (cX, cY)
        return centroidPos
    else:
        return None
    
def getAllFilteredRoi(id, frame, filterMaxVal = 0, withinRegion = None):
    roidFrame = frame.copy()
    filtered_image = frame.copy()

    if gvars.glist_rois:
        depthFilteredFrame = gvars.glist_rois[id].getDepthFilteredFrame(frame)
        roidFrame = gvars.glist_rois[id].getRoiMaskedFrame(depthFilteredFrame)

        if withinRegion:
            mask = np.zeros_like(roidFrame, dtype=np.uint8)
            x1, y1, x2, y2 = withinRegion

            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1) 

            roidFrame = cv2.bitwise_and(roidFrame, roidFrame, mask=mask)

        if filterMaxVal != 0:
            max_depth_value = np.max(roidFrame)
            threshold_value = max_depth_value - (max_depth_value/gvars.glist_rois[id].maxValFilterFactor) # - to do: transformar esse valor em variavel do Roi

            # Threshold the depth image
            _, mask = cv2.threshold(roidFrame, threshold_value, 255, cv2.THRESH_BINARY)

            # Apply the mask to the original filtered image
            filtered_image = cv2.bitwise_and(depthFilteredFrame, depthFilteredFrame, mask=mask)
        else:
            filtered_image = roidFrame

    return filtered_image

# DRAW funcs---------------------------------------------------------------------------------------------------
def drawRoiPoints(event, x, y, flags, param):
    selectedRoi = gvars.g_selectedRoi

    if event == cv2.EVENT_LBUTTONDOWN:
        gvars.glist_rois[selectedRoi].l_points.append((x, y))
    
    elif event == cv2.EVENT_RBUTTONDOWN:
        if len(gvars.glist_rois[selectedRoi].l_points) > 0:
            gvars.glist_rois[selectedRoi].l_points.pop()

def drawRoisOutlines(selectedRoi, frame):
    for roi in gvars.glist_rois:
        flag = True
        for point in roi.l_points:
            cv2.circle(frame, point, 3, (200, 200, 200), 4)
            if flag:
                flag = False
                color = (200, 200, 200)
                if roi.id == selectedRoi:
                    color = gvars.roiSelectedColor
                cv2.putText(frame, str(roi.id), point, cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 1, cv2.LINE_AA)
        if len(roi.l_points) > 1:
            color = (120, 120, 120)
            if roi.id == selectedRoi:
                color = gvars.roiSelectedColor
            cv2.polylines(frame, [np.array(roi.l_points)], False, color, thickness=2)

def drawAlreadyFoundRegionsAndCentroids(regions, centroids, frame, color = (180, 0, 180), thickness = 1):
    i = 0
    for region in regions:
        x1, y1, x2, y2 = region
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        if centroids[i] is not None:
            cv2.circle(frame, (centroids[i][0] + x1, centroids[i][1] + y1), thickness, color, thickness)
        cv2.putText(frame, str(i), (x1 - 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, color, thickness, cv2.LINE_AA)

        i += 1

def drawBrightestRegions(id, frameA, frameD):
    if gvars.glist_rois:
        brightestRegions = gvars.glist_rois[id].findBrightestRegions(frameA, gvars.glist_rois[id].numBrightestRegions, gvars.glist_rois[id].overRegionsFactor)

        i = 0
        colorRect = (0, 255, 0) # brightest
        sizeRect = 2 # brightest
        colorCentroid = (0, 255, 0) # brightest
        sizeCentroid = 3

        if len(brightestRegions) > 0:
            for region in brightestRegions:
                i += 1

                x1, y1, x2, y2 = region
                cv2.rectangle(frameD, (x1, y1), (x2, y2), colorRect, sizeRect)

                region = frameA[y1:y2, x1:x2]

                centroid = computeCentroid(region)

                if centroid is not None:
                    cv2.circle(frameD, (centroid[0] + x1, centroid[1] + y1), 3, colorCentroid, sizeCentroid)
                if i == 1:
                    colorRect = 255
                    sizeRect = 0
                    colorCentroid = (200, 200, 200)
                    sizeCentroid = 0

def getRegionsCentroid(frameA, l_regions):
    l_centroids = []

    if len(l_regions) > 0:
        for region in l_regions:

            x1, y1, x2, y2 = region

            region = frameA[y1:y2, x1:x2]

            centroid = computeCentroid(region)

            #if centroid is not None:
            #    l_centroids.append(centroid)
            l_centroids.append(centroid)
    
    return l_centroids

def findClosestPoint(centroids, regions, targetCentroidsAbs, maxDistance = 50):
    if not centroids or not regions or len(centroids) != len(regions):
        raise ValueError("Centroids and regions must be non-empty lists of the same length.")
    
    # Convert relative coordinates to absolute coordinates for the main centroids
    centroidsAbsolute = np.array([
        (centroid[0] + region[0], centroid[1] + region[1]) 
        for centroid, region in zip(centroids, regions)
        if centroid is not None
    ])
    
    # Compute the pairwise distances between all target centroids and all main centroids
    distances = np.linalg.norm(centroidsAbsolute[:, np.newaxis, :] - targetCentroidsAbs[np.newaxis, :, :], axis=2)
    
    # Find the index of the minimum distance
    minIdx = np.unravel_index(np.argmin(distances), distances.shape)

    # Get the minimum distance
    minDistance = distances[minIdx]
    
    # Get the closest centroid and region
    closestCentroidIndex = minIdx[0]
    closestRegion = regions[closestCentroidIndex]
    closestCentroid = centroids[closestCentroidIndex]

    noCloseRegionsFound = False

    if maxDistance is not None and minDistance >= maxDistance:
        noCloseRegionsFound = True
    
    return closestRegion, closestCentroid, closestCentroidIndex, noCloseRegionsFound

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








# ---- funcoes inutilizadas por hora

def getOverlapBetweenTwoRegions(region1, region2):
    x1_min, y1_min, x1_max, y1_max = region1
    x2_min, y2_min, x2_max, y2_max = region2

    x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
    y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))

    return x_overlap * y_overlap

def isRegionClustered(regionsToSearch, mainRegionId, overlapThreshold = 0, minRegionsForCluster = 2):
    if overlapThreshold == 0:
        overlapThreshold = gvars.glist_rois[mainRegionId].overRegionsFactor - (gvars.glist_rois[mainRegionId].overRegionsFactor / 10)
        
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