# globalVars
frameShape = None
l_rois = []
l_itensToTrack = []
curPedal = -1
curItem = 0
selectedRoiId = None
virginFrame = None
filterPerBrightest = True

moveorrotate = 0

maxTimeChangingRoi = 4
maxTimeNotFindingCloseRegion = 10
lastCentroidsForPredictionNum = 20

itemTypeDictionary = {
    "r": "rh",
    "l": "lh",
    "v": "vl"
}

itemTrackingLoop = False

roiSelectedColor = (130, 250, 250)
regionVal = 10

l_rightHandRois = [
    0, #p0
    1, #p1
    2, #p2
    6, #p3-
]

l_leftHandRois = [
    3, 4, #p2
    5, #p3
    6, #p4
]

test = False