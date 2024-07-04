import cv2
import numpy as np
import pickle
from pythonosc import udp_client
import _1_classes as classes
import _2_funcs as funcs
import _3_gvars as gvars

ip = "127.0.0.1"
port = 8000  
client = udp_client.SimpleUDPClient(ip, port)

gvars.l_itensToTrack.append(classes.ItemToTrack("r", [0]))
gvars.l_itensToTrack.append(classes.ItemToTrack("l", [2]))

try:
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        raise Exception("Error: Camera not accessible.")
    
    cv2.namedWindow('Roi')
    cv2.setMouseCallback('Roi', funcs.drawRoiPoints)
    cv2.createTrackbar('minDepth', 'Roi', 70, 255, funcs.update_minDepth)
    cv2.createTrackbar('maxDepth', 'Roi', 130, 255, funcs.update_maxDepth)
    cv2.createTrackbar('maxValFilterFactor', 'Roi', 300, 1000, funcs.update_maxValFilterFactor)
    cv2.createTrackbar('brightestSizeX', 'Roi', 30, 100, funcs.update_brightestRegionX)
    cv2.createTrackbar('brightestSizeY', 'Roi', 30, 100, funcs.update_brightestRegionY)
    cv2.createTrackbar('numBrightestRegions', 'Roi', 1, 10, funcs.update_numBrightestRegions)
    cv2.createTrackbar('overRegionsFactor', 'Roi', 25, 100, funcs.update_overRegionsFactor)
    cv2.createTrackbar('minDistBetweenPoints', 'Roi', 40, 100, funcs.update_minDistBetweenPoints)

    cv2.namedWindow('SelectedRoiDepthFiltered')

    while True:
        # Read a frame from the camera
        success, gvars.virginFrame = cap.read()

        if not success:
            raise Exception("Error: Failed to read frame from camera.")
        
        frame = gvars.virginFrame.copy()
        l_alreadyRenderedRois = []

        # vfilteredFrame = funcs.getAllFilteredRoi(gvars.g_selectedRoi, frame)
        # # Normalize the filtered frame for display purposes
        # vnormalized_frame = cv2.normalize(vfilteredFrame, None, 0, 255, cv2.NORM_MINMAX)
        # vnormalized_frame = np.uint8(vnormalized_frame)

        # cv2.putText(gvars.virginFrame, str(gvars.g_selectedRoi), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, gvars.roiSelectedColor, 2, cv2.LINE_AA)

        # funcs.drawRoisOutlines(gvars.g_selectedRoi, gvars.virginFrame)
        # funcs.drawTest(gvars.g_selectedRoi, vnormalized_frame, gvars.virginFrame)
        
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        clahe_frame = clahe.apply(frame_gray)

        height, width = frame.shape[:2]
        allRois_frame = np.zeros((height, width), dtype=np.uint8)

        if gvars.itemTrackingLoop:
            for item in gvars.l_itensToTrack:
                if item.isActive:   
                    if not item.l_orderedRois[item.curRoi] in l_alreadyRenderedRois:
                        clahe_filteredFrame = []
                        roied_clahe_normalized_frame = []

                        allRois_frame = funcs.getAllFilteredRoi(item.l_orderedRois[item.curRoi], clahe_frame, gvars.filterPerBrightest)
                        roied_clahe_normalized_frame = cv2.normalize(allRois_frame, None, 0, 255, cv2.NORM_MINMAX)
                        roied_clahe_normalized_frame = np.uint8(roied_clahe_normalized_frame)

                    if item.isBrightestForSure:
                        l_brightestRegions = gvars.glist_rois[item.l_orderedRois[item.curRoi]].findBrightestRegions(roied_clahe_normalized_frame, 1)

                        if (l_brightestRegions is not None) & (l_brightestRegions is not []):
                            brightestRegion = l_brightestRegions[0]

                            x1, y1, x2, y2 = brightestRegion
                            region = roied_clahe_normalized_frame[y1:y2, x1:x2]
                            centroid = funcs.computeCentroid(region)

                            item.l_lastRegionFound = brightestRegion
                            item.lastRegionCentroid = centroid
                            item.lastRegionAsBasis = True
                            
                            funcs.drawAlreadyFoundRegionsAndCentroids(l_brightestRegions, [centroid], gvars.virginFrame)
                    elif (item.isBrightestForSure is not True) & (item.lastRegionAsBasis):
                        l_brightestRegions = gvars.glist_rois[item.l_orderedRois[item.curRoi]].findBrightestRegions(roied_clahe_normalized_frame, gvars.glist_rois[item.l_orderedRois[item.curRoi]].numBrightestRegions + item.timeNotFindingCloseRegion, gvars.glist_rois[item.l_orderedRois[item.curRoi]].overRegionsFactor)                     
                        l_centroids = []

                        if len(l_brightestRegions) > 0:  
                            l_centroids = funcs.getRegionsCentroid(roied_clahe_normalized_frame, l_brightestRegions)
                            lastCentroidAbs = (item.lastRegionCentroid[0] + item.l_lastRegionFound[0], item.lastRegionCentroid[1] + item.l_lastRegionFound[1]) 

                            # busca o ponto mais proximo da regiao identificada ou das ultimas predicoes de regioes
                            closestBrightestRegion, closestBrightestRegionCentroid, closestRegionCentroidId, noCloseRegionFound = funcs.findClosestPoint(l_centroids, l_brightestRegions, np.array([lastCentroidAbs] + item.l_predictedCentroidsAbs))
                            
                            color = (0, 0, 255)

                            if(item.isChangingRoi):
                                color = (180, 255, 180)

                            # se nao encontrar regiao perto o suficiente
                            if (noCloseRegionFound) & (item.timeNotFindingCloseRegion <= gvars.maxTimeNotFindingCloseRegion) & (item.isChangingRoi is not True):
                                x1, y1, x2, y2 = l_brightestRegions[closestRegionCentroidId]
                                color = (255, 0, 0)

                                # prediz a posicao onde o item deveria estar
                                predictedCentroidAbs = funcs.predictNextCentroid(item.l_lastCentroidsAbs)
                                item.l_predictedCentroidsAbs.append(predictedCentroidAbs)

                                for predictedCentroid in item.l_predictedCentroidsAbs:
                                    cv2.circle(gvars.virginFrame, (predictedCentroid[0], predictedCentroid[1]), 3, (190, 190, 190), 2)
                                    
                                item.timeNotFindingCloseRegion += 1
                            else: # se encontrar, 
                                x1, y1, x2, y2 = l_brightestRegions[closestRegionCentroidId]

                                item.l_lastRegionFound = closestBrightestRegion
                                item.lastRegionCentroid = closestBrightestRegionCentroid

                                item.timeNotFindingCloseRegion = 0
                                item.l_predictedCentroidsAbs = []

                            funcs.drawAlreadyFoundRegionsAndCentroids(l_brightestRegions, l_centroids, gvars.virginFrame)
                            
                            x1, y1, x2, y2 = item.l_lastRegionFound
                            cv2.rectangle(gvars.virginFrame, (x1, y1), (x2, y2), color, 4)
                            cv2.circle(gvars.virginFrame, (item.lastRegionCentroid[0] + x1, item.lastRegionCentroid[1] + y1), 3, color, 4)
                                        
                            if item.type == "r":
                                x1, y1, x2, y2 = item.l_lastRegionFound
                                cv2.putText(gvars.virginFrame, gvars.itemTypeDictionary[item.type], (x2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (180, 180, 255), 2, cv2.LINE_AA)
                            
                                if item.isProjecting:
                                    client.send_message("/rh/pointX", int(closestBrightestRegionCentroid[0] + x1))
                                    client.send_message("/rh/pointY", int(closestBrightestRegionCentroid[1] + y1))
                            elif item.type == "l":
                                x1, y1, x2, y2 = item.l_lastRegionFound
                                cv2.putText(gvars.virginFrame, gvars.itemTypeDictionary[item.type], (x2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (180, 180, 255), 2, cv2.LINE_AA)
                            
                                if item.isProjecting:
                                    client.send_message("/lh/pointX", int(closestBrightestRegionCentroid[0] + x1))
                                    client.send_message("/lh/pointY", int(closestBrightestRegionCentroid[1] + y1))
                            elif item.type == "v":
                                x1, y1, x2, y2 = item.l_lastRegionFound
                                cv2.putText(gvars.virginFrame, gvars.itemTypeDictionary[item.type], (x2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (180, 180, 255), 2, cv2.LINE_AA)
                            
                                if item.isProjecting:
                                    client.send_message("/vl/pointX", int(closestBrightestRegionCentroid[0] + x1))
                                    client.send_message("/vl/pointY", int(closestBrightestRegionCentroid[1] + y1))

                            if item.isChangingRoi:
                                if item.timeChangingRoi < gvars.maxTimeChangingRoi:
                                    item.timeChangingRoi += 1
                                else:
                                    item.timeChangingRoi = 0
                                    item.isChangingRoi = False
                                    
                            if len(item.l_lastCentroidsAbs) < gvars.lastCentroidsForPredictionNum:
                                x1, y1, x2, y2 = item.l_lastRegionFound
                                centroidAbs = (item.lastRegionCentroid[0] + x1, item.lastRegionCentroid[1] + y1)
                                item.l_lastCentroidsAbs.append(centroidAbs)
                            else:
                                item.l_lastCentroidsAbs.pop(0)
                                x1, y1, x2, y2 = item.l_lastRegionFound
                                centroidAbs = (item.lastRegionCentroid[0] + x1, item.lastRegionCentroid[1] + y1)
                                item.l_lastCentroidsAbs.append(centroidAbs)

                    
                    l_alreadyRenderedRois.append(item.l_orderedRois[item.curRoi])
                            
        else: # se nao tive items a procurar, desenha regioes relacionadas ao roi selecionado
            allRois_frame = funcs.getAllFilteredRoi(gvars.g_selectedRoi, clahe_frame, gvars.filterPerBrightest)
            cnormalized_frame = cv2.normalize(allRois_frame, None, 0, 255, cv2.NORM_MINMAX)
            cnormalized_frame = np.uint8(cnormalized_frame)

            funcs.drawBrightestRegions(gvars.g_selectedRoi, cnormalized_frame, gvars.virginFrame)

        # desenha contornos dos Rois
        funcs.drawRoisOutlines(gvars.g_selectedRoi, gvars.virginFrame)
        cv2.putText(gvars.virginFrame, str(gvars.g_selectedRoi), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, gvars.roiSelectedColor, 2, cv2.LINE_AA)

        for item in gvars.l_itensToTrack:
            if item.isActive & item.lastRegionAsBasis:
                x1, y1, x2, y2 = item.l_lastRegionFound
                cv2.rectangle(clahe_frame, (x1, y1), (x2, y2), (180, 0, 180), 1)
                cv2.circle(clahe_frame, (item.lastRegionCentroid[0] + x1, item.lastRegionCentroid[1] + y1), 3, (220, 220, 0), 1)
                cv2.putText(clahe_frame, str(item.type), (x1 - 10, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, gvars.roiSelectedColor, 2, cv2.LINE_AA)

                cv2.rectangle(allRois_frame, (x1, y1), (x2, y2), (180, 0, 180), 1)
                cv2.circle(allRois_frame, (item.lastRegionCentroid[0] + x1, item.lastRegionCentroid[1] + y1), 3, (220, 220, 0), 1)
                cv2.putText(allRois_frame, str(item.type), (x1 - 10, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, gvars.roiSelectedColor, 2, cv2.LINE_AA)

        cv2.imshow('clahe - preprocessedImage', clahe_frame)
        cv2.imshow('SelectedRoiDepthFiltered', allRois_frame)
        cv2.imshow('Roi', gvars.virginFrame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'): 
            break
        elif key in range(ord('1'), ord('9')):  
            keyIntVal = int(chr(key))
            
            if(len(gvars.glist_rois) >= keyIntVal):
                funcs.selectRoi(keyIntVal - 1)
            else:
                print(f"somente {len(gvars.glist_rois)} rois existem. Criando mais um...")
                gvars.glist_rois.append(classes.Roi(len(gvars.glist_rois)))
                gvars.g_selectedRoi = len(gvars.glist_rois) - 1
        elif key == ord('d'):
            gvars.glist_rois[gvars.g_selectedRoi].eraseAllPoints()
        elif key == ord('p'):
            with open('savedRois.pkl', 'wb') as file:
                pickle.dump(gvars.glist_rois, file)
                print(gvars.glist_rois)
                print("saved rois file--------------------")
        elif key == ord('ç'):
            with open('savedRois.pkl', 'rb') as file:
                gvars.glist_rois = pickle.load(file)
                print("loaded rois file--------------------")

                if len(gvars.glist_rois) > 0:
                    gvars.g_selectedRoi = 0
        elif key == ord(' '):
            gvars.itemTrackingLoop = not gvars.itemTrackingLoop 
        elif key == ord('y'): # rightHand
            if len(gvars.l_itensToTrack) >= 1:
                gvars.l_itensToTrack[0].isActive = not gvars.l_itensToTrack[0].isActive
        elif key == ord('u'): # leftHand
            if len(gvars.l_itensToTrack) >= 2:
                gvars.l_itensToTrack[1].isActive = not gvars.l_itensToTrack[1].isActive
        elif key == ord('i'): # voluta
            if len(gvars.l_itensToTrack) >= 3:
                gvars.l_itensToTrack[2].isActive = not gvars.l_itensToTrack[2].isActive
        elif key == ord('h'): # rightHand
            if len(gvars.l_itensToTrack) >= 1:
                gvars.l_itensToTrack[0].isBrightestForSure = not gvars.l_itensToTrack[0].isBrightestForSure
        elif key == ord('j'): # leftHand
            if len(gvars.l_itensToTrack) >= 2:
                gvars.l_itensToTrack[1].isBrightestForSure = not gvars.l_itensToTrack[1].isBrightestForSure
        elif key == ord('k'): # voluta
            if len(gvars.l_itensToTrack) >= 3:
                gvars.l_itensToTrack[2].isBrightestForSure = not gvars.l_itensToTrack[2].isBrightestForSure
        elif key == ord('b'): 
            gvars.filterPerBrightest = not gvars.filterPerBrightest
        elif key == ord('w'): 
            if len(gvars.l_itensToTrack) >= 1:
                gvars.l_itensToTrack[0].l_orderedRois = [1]
                gvars.l_itensToTrack[0].isChangingRoi = True
        elif key == ord('e'): 
            if len(gvars.l_itensToTrack) >= 1:
                gvars.l_itensToTrack[0].l_orderedRois = [0]
        elif key == ord('r'): 
            if len(gvars.l_itensToTrack) >= 2:
                gvars.l_itensToTrack[1].l_orderedRois = [1]
        elif key == ord('t'): 
            if len(gvars.l_itensToTrack) >= 2:
                gvars.l_itensToTrack[1].l_orderedRois = [2]
        elif key == ord('z'): 
            if len(gvars.l_itensToTrack) >= 1:
                gvars.l_itensToTrack[0].isProjecting = not gvars.l_itensToTrack[0].isProjecting
                client.send_message("/rh/isDrawing", int(gvars.l_itensToTrack[0].isProjecting))
        elif key == ord('x'): 
            if len(gvars.l_itensToTrack) >= 2:
                gvars.l_itensToTrack[1].isProjecting = not gvars.l_itensToTrack[1].isProjecting
                client.send_message("/lh/isDrawing", int(gvars.l_itensToTrack[1].isProjecting))
        elif key == ord('c'): 
            if len(gvars.l_itensToTrack) >= 3:
                gvars.l_itensToTrack[2].isProjecting = not gvars.l_itensToTrack[2].isProjecting
                client.send_message("/vl/isDrawing", int(gvars.l_itensToTrack[2].isProjecting))

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    # Release the video capture object
    if 'cap' in locals() and cap.isOpened():
        cap.release()
    
    # Close all OpenCV windows
    cv2.destroyAllWindows()