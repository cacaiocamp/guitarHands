import cv2
import numpy as np
import pickle
from pynput import keyboard
from pythonosc import udp_client
import _1_classes as classes
import _2_funcs as funcs
import _3_gvars as gvars
import _4_pedals as pedals

ip = "127.0.0.1"
port = 8000  
client = udp_client.SimpleUDPClient(ip, port)

gvars.l_itensToTrack.append(classes.ItemToTrack("r", gvars.l_rightHandRois))
gvars.l_itensToTrack.append(classes.ItemToTrack("l", gvars.l_leftHandRois))

arrow_keys = {
    'left': False,
    'up': False,
    'right': False,
    'down': False
}

# Define callback functions for key press and release
def on_press(akey):
    try:
        if akey == keyboard.Key.left:
            arrow_keys['left'] = True
        elif akey == keyboard.Key.up:
            arrow_keys['up'] = True
        elif akey == keyboard.Key.right:
            arrow_keys['right'] = True
        elif akey == keyboard.Key.down:
            arrow_keys['down'] = True
    except AttributeError:
        pass

def on_release(akey):
    try:
        if akey == keyboard.Key.left:
            arrow_keys['left'] = False
        elif akey == keyboard.Key.up:
            arrow_keys['up'] = False
        elif akey == keyboard.Key.right:
            arrow_keys['right'] = False
        elif akey == keyboard.Key.down:
            arrow_keys['down'] = False
    except AttributeError:
        pass

listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

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
    cv2.createTrackbar('maxDistanceBetweenPoints', 'Roi', 40, 100, funcs.update_maxDistanceBetweenPoints)

    cv2.namedWindow('SelectedRoiDepthFiltered')

    while True:
        # Read a frame from the camera
        success, gvars.virginFrame = cap.read()

        if not success:
            raise Exception("Error: Failed to read frame from camera.")
        
        frame = gvars.virginFrame.copy()
        
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        clahe_frame = clahe.apply(frame_gray)

        height, width = frame.shape[:2]
        gvars.frameShape = (width, height)
        allRois_frame = np.zeros((height, width), dtype=np.uint8)

        l_alreadyRenderedRois = []

        if gvars.itemTrackingLoop:
            curItemId = 0
            for item in gvars.l_itensToTrack:
                if item.isActive:   
                    if not item.l_orderedRois[item.curRoi] in l_alreadyRenderedRois:
                        clahe_filteredFrame = []
                        roied_clahe_normalized_frame = []

                        allRois_frame = funcs.getAllFilteredRoi(item.l_orderedRois[item.curRoi], clahe_frame, gvars.filterPerBrightest)
                        roied_clahe_normalized_frame = cv2.normalize(allRois_frame, None, 0, 255, cv2.NORM_MINMAX)
                        roied_clahe_normalized_frame = np.uint8(roied_clahe_normalized_frame)
                        
                        l_brightestRegions = gvars.l_rois[item.l_orderedRois[item.curRoi]].findBrightestRegions(
                            roied_clahe_normalized_frame, 
                            gvars.l_rois[item.l_orderedRois[item.curRoi]].numBrightestRegions,
                            gvars.l_rois[item.l_orderedRois[item.curRoi]].overRegionsFactor
                        )
                        
                    if item.isBrightestForSure:
                        if (gvars.l_rois[item.l_orderedRois[item.curRoi]].l_brightestRegionsFound is not None) & (gvars.l_rois[item.l_orderedRois[item.curRoi]].l_brightestRegionsFound is not []):      
                            item.lastRegionFoundAsItem = gvars.l_rois[item.l_orderedRois[item.curRoi]].l_brightestRegionsFound[0]
                            item.lastRegionAsBasis = True
                            
                            funcs.drawAlreadyFoundRegionsAndCentroids(gvars.virginFrame, gvars.l_rois[item.l_orderedRois[item.curRoi]].l_brightestRegionsFound)

                    elif (item.isBrightestForSure is not True) & (item.lastRegionAsBasis):
                        if len(gvars.l_rois[item.l_orderedRois[item.curRoi]].l_brightestRegionsFound) > 0:  
                            lastCentroidAbs = item.lastRegionFoundAsItem.getCentroidAbsolutePosition()

                            # busca o ponto mais proximo da regiao identificada e das ultimas predicoes de regioes
                            closestBrightestRegion, closestBrightestRegionId, noCloseRegionFound = funcs.findClosestPoint(gvars.l_rois[item.l_orderedRois[item.curRoi]].l_brightestRegionsFound, np.array([lastCentroidAbs] + item.l_predictedCentroidsAbs), gvars.l_rois[item.l_orderedRois[item.curRoi]].maxDistanceBetweenPoints)
                            
                            color = (0, 0, 255)
                            if(item.isChangingRoi):
                                color = (180, 255, 180)

                            # se nao encontrar regiao perto o suficiente
                            if (noCloseRegionFound) & (item.timeNotFindingCloseRegion <= gvars.maxTimeNotFindingCloseRegion) & (item.isChangingRoi is not True):
                                color = (255, 0, 0)

                                # prediz a posicao onde o item deveria estar
                                predictedCentroidAbs = funcs.predictNextCentroid(item.l_lastCentroidsAbs)
                                item.l_predictedCentroidsAbs.append(predictedCentroidAbs)

                                for predictedCentroid in item.l_predictedCentroidsAbs:
                                    cv2.circle(gvars.virginFrame, (predictedCentroid[0], predictedCentroid[1]), 3, (190, 190, 190), 2)
                                    
                                item.timeNotFindingCloseRegion += 1

                                gvars.l_rois[item.l_orderedRois[item.curRoi]].findBrightestRegions(
                                    roied_clahe_normalized_frame, 
                                    gvars.l_rois[item.l_orderedRois[item.curRoi]].numBrightestRegions + item.timeNotFindingCloseRegion,
                                    gvars.l_rois[item.l_orderedRois[item.curRoi]].overRegionsFactor,
                                    True #ja tinha encontrado as brightest regions nesse frame para esse roi
                                )

                            else: # se encontrar, 
                                item.lastRegionFoundAsItem = closestBrightestRegion
                                item.lastRegionFoundAsItemId = closestBrightestRegionId
                                gvars.l_rois[item.l_orderedRois[item.curRoi]].l_brightestRegionsFound[closestBrightestRegionId].selectedAsItem = True

                                item.timeNotFindingCloseRegion = 0
                                item.l_predictedCentroidsAbs = []

                            item.lastRegionFoundAsItem.draw(gvars.virginFrame, color, 4, color, 4)

                            if item.overlapJumpToNextRoi:
                                item.checkItemInNextRoiOverlap()
                                item.getImageFrameWithRoiOverlap(gvars.virginFrame)

                            if item.type == "r":
                                x1, y1, x2, y2 = item.lastRegionFoundAsItem.l_points
                                cv2.putText(gvars.virginFrame, gvars.itemTypeDictionary[item.type], (x2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (180, 180, 255), 2, cv2.LINE_AA)
                            
                                if item.isProjecting:
                                    client.send_message("/rh/pointX", int(item.lastRegionFoundAsItem.centroid[0] + x1))
                                    client.send_message("/rh/pointY", int(item.lastRegionFoundAsItem.centroid[1] + y1))
                            elif item.type == "l":
                                x1, y1, x2, y2 = item.lastRegionFoundAsItem.l_points
                                cv2.putText(gvars.virginFrame, gvars.itemTypeDictionary[item.type], (x2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (180, 180, 255), 2, cv2.LINE_AA)
                            
                                if item.isProjecting:
                                    client.send_message("/lh/pointX", int(item.lastRegionFoundAsItem.centroid[0] + x1))
                                    client.send_message("/lh/pointY", int(item.lastRegionFoundAsItem.centroid[1] + y1))
                            elif item.type == "v":
                                x1, y1, x2, y2 = item.lastRegionFoundAsItem.l_points
                                cv2.putText(gvars.virginFrame, gvars.itemTypeDictionary[item.type], (x2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (180, 180, 255), 2, cv2.LINE_AA)
                            
                                if item.isProjecting:
                                    client.send_message("/vl/pointX", int(item.lastRegionFoundAsItem.centroid[0] + x1))
                                    client.send_message("/vl/pointY", int(item.lastRegionFoundAsItem.centroid[1] + y1))

                            if item.isChangingRoi:
                                if item.timeChangingRoi < gvars.maxTimeChangingRoi:
                                    item.timeChangingRoi += 1
                                else:
                                    item.timeChangingRoi = 0
                                    item.isChangingRoi = False
                                    
                            if len(item.l_lastCentroidsAbs) < gvars.lastCentroidsForPredictionNum:
                                x1, y1, x2, y2 = item.lastRegionFoundAsItem.l_points
                                centroidAbs = item.lastRegionFoundAsItem.getCentroidAbsolutePosition()
                                item.l_lastCentroidsAbs.append(centroidAbs)
                            else:
                                item.l_lastCentroidsAbs.pop(0)
                                x1, y1, x2, y2 = item.lastRegionFoundAsItem.l_points
                                centroidAbs = item.lastRegionFoundAsItem.getCentroidAbsolutePosition()
                                item.l_lastCentroidsAbs.append(centroidAbs)
                    
                    funcs.drawAlreadyFoundRegionsAndCentroids(gvars.virginFrame, gvars.l_rois[item.l_orderedRois[item.curRoi]].l_brightestRegionsFound)

                    l_alreadyRenderedRois.append(item.l_orderedRois[item.curRoi])
                
                curItemId += 1
                            
        else: # se nao tiver items a procurar, desenha regioes relacionadas ao roi selecionado
            if gvars.selectedRoiId is not None:
                allRois_frame = funcs.getAllFilteredRoi(gvars.selectedRoiId, clahe_frame, gvars.filterPerBrightest)
                cnormalized_frame = cv2.normalize(allRois_frame, None, 0, 255, cv2.NORM_MINMAX)
                cnormalized_frame = np.uint8(cnormalized_frame)

                gvars.l_rois[gvars.selectedRoiId].findBrightestRegions(
                    cnormalized_frame, 
                    gvars.l_rois[gvars.selectedRoiId].numBrightestRegions,
                    gvars.l_rois[gvars.selectedRoiId].overRegionsFactor
                )

                funcs.drawBrightestRegions(gvars.virginFrame, gvars.l_rois[gvars.selectedRoiId].l_brightestRegionsFound)

        # desenha contornos dos Rois
        funcs.drawRoisOutlines(gvars.virginFrame, gvars.selectedRoiId, )
        cv2.putText(gvars.virginFrame, str(gvars.selectedRoiId), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, gvars.roiSelectedColor, 2, cv2.LINE_AA)

        for item in gvars.l_itensToTrack:
            if item.isActive & item.lastRegionAsBasis:
                x1, y1, x2, y2 = item.lastRegionFoundAsItem.l_points

                item.lastRegionFoundAsItem.draw(clahe_frame, (180, 0, 180), 1, (220, 220, 0), 3)
                cv2.putText(clahe_frame, str(item.type), (x1 - 10, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, gvars.roiSelectedColor, 2, cv2.LINE_AA)

                item.lastRegionFoundAsItem.draw(allRois_frame, (180, 0, 180), 1, (220, 220, 0), 3)
                cv2.putText(allRois_frame, str(item.type), (x1 - 10, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, gvars.roiSelectedColor, 2, cv2.LINE_AA)

        cv2.imshow('clahe - preprocessedImage', clahe_frame)
        cv2.imshow('SelectedRoiDepthFiltered', allRois_frame)
        cv2.imshow('Roi', gvars.virginFrame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'): 
            break

        # Rois Actions
        elif key == ord('a'): # add Roi
            gvars.l_rois.append(classes.Roi(len(gvars.l_rois)))
            gvars.selectedRoiId = len(gvars.l_rois) - 1
        elif (key == ord('s')) | (key == ord('+')): # change selected Roi +
            nextRoi = gvars.selectedRoiId + 1
            if(nextRoi == len(gvars.l_rois)):
                nextRoi = 0
            gvars.selectedRoiId = nextRoi
            funcs.selectRoi(gvars.selectedRoiId)
        elif key == ord('-'): # change selected Roi -
            nextRoi = gvars.selectedRoiId - 1
            if(nextRoi == -1):
                nextRoi = len(gvars.l_rois) - 1
            gvars.selectedRoiId = nextRoi
            funcs.selectRoi(gvars.selectedRoiId)
        elif key == ord('d'): # delete Roi points
            gvars.l_rois[gvars.selectedRoiId].eraseAllPoints()
        elif key == ord('p'): # save rois in pickle
            with open('savedRois.pkl', 'wb') as file:
                pickle.dump(gvars.l_rois, file)
                print(gvars.l_rois)
                print("saved rois file--------------------")
        elif key == ord('ç'): # load rois from pickle
            with open('savedRois.pkl', 'rb') as file:
                gvars.l_rois = pickle.load(file)
                print("loaded rois file--------------------")

                if len(gvars.l_rois) > 0:
                    gvars.selectedRoiId = 0
        #--------------------------------------------------
        elif key == ord('*'):
            gvars.curPedal = gvars.curPedal + 1
            pedals.callPedalEvent(gvars.curPedal)
        elif key == ord('/'):
            print('a')
        elif key == ord('r'): 
            gvars.moveorrotate = not gvars.moveorrotate


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
        elif key == ord('x'): # rightHand
            if len(gvars.l_itensToTrack) >= 1:
                gvars.curItem = 0
        elif key == ord('c'): # leftHand
            if len(gvars.l_itensToTrack) >= 2:
                gvars.curItem = 1
        elif key == ord('v'): # voluta
            if len(gvars.l_itensToTrack) >= 3:
                gvars.curItem = 2
        elif key in range(ord('1'), ord('4')):  
            gvars.curItem = 1
            keyIntVal = int(chr(key)) - 1
            
            curRoi = gvars.l_itensToTrack[gvars.curItem].l_orderedRois[gvars.l_itensToTrack[gvars.curItem].curRoi]

            if keyIntVal < len(gvars.l_rois[curRoi].l_brightestRegionsFound):
                gvars.l_rois[curRoi].l_brightestRegionsFound[keyIntVal]

                gvars.l_itensToTrack[gvars.curItem].lastRegionFoundAsItem = gvars.l_rois[curRoi].l_brightestRegionsFound[keyIntVal]
                gvars.l_itensToTrack[gvars.curItem].lastRegionFoundAsItemId = keyIntVal
                gvars.l_rois[curRoi].l_brightestRegionsFound[keyIntVal].selectedAsItem = True

                item.timeNotFindingCloseRegion = 0
                item.l_predictedCentroidsAbs = []
        elif key in range(ord('4'), ord('7')):  
            gvars.curItem = 0
            keyIntVal = int(chr(key)) - 4
            
            curRoi = gvars.l_itensToTrack[gvars.curItem].l_orderedRois[gvars.l_itensToTrack[gvars.curItem].curRoi]

            if keyIntVal < len(gvars.l_rois[curRoi].l_brightestRegionsFound):
                gvars.l_rois[curRoi].l_brightestRegionsFound[keyIntVal]

                gvars.l_itensToTrack[gvars.curItem].lastRegionFoundAsItem = gvars.l_rois[curRoi].l_brightestRegionsFound[keyIntVal]
                gvars.l_itensToTrack[gvars.curItem].lastRegionFoundAsItemId = keyIntVal
                gvars.l_rois[curRoi].l_brightestRegionsFound[keyIntVal].selectedAsItem = True

                item.timeNotFindingCloseRegion = 0
                item.l_predictedCentroidsAbs = []
        
        if arrow_keys['left']:
            if gvars.moveorrotate == 0:
                funcs.moveAllRois(-1, 0)
            elif gvars.moveorrotate == 1:
                funcs.rotateAllRois(-1, (int(width/2), int(height/2)))
        if arrow_keys['up']:
            funcs.moveAllRois(0, -1)
        if arrow_keys['right']:
            if gvars.moveorrotate == 0:
                funcs.moveAllRois(1, 0)
            elif gvars.moveorrotate == 1:
                funcs.rotateAllRois(1, (int(width/2), int(height/2)))
        if arrow_keys['down']:
            funcs.moveAllRois(0, 1)

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    # Release the video capture object
    if 'cap' in locals() and cap.isOpened():
        cap.release()
    
    # Close all OpenCV windows
    cv2.destroyAllWindows()
    listener.stop()