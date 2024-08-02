import cv2
import numpy as np
import _3_gvars as gvars

def callPedalEvent(newPedal):
    if(newPedal == 0):
        pedal0()
    elif(newPedal == 1):
        pedal1()
    elif(newPedal == 2):
        pedal1emeio()
    elif(newPedal == 3):
        pedal2()
    elif(newPedal == 4):
        pedal3()
    elif(newPedal == 5):
        pedal4()


def pedal0():
    print("--- pedal 0 --------------------")
    gvars.l_itensToTrack[0].curRoi = 0
    gvars.l_itensToTrack[0].isActive = True
    gvars.l_itensToTrack[0].isBrightestForSure = True
def pedal1():
    print("--- pedal 1 --------------------")
    # ativa troca por sobreposicao para roi 1 na MD
    gvars.l_itensToTrack[0].allowOverlapRoiChange()
    gvars.l_itensToTrack[0].isProjecting = True
def pedal1emeio():
    print("--- pedal 1 e meio --------------------")
    # assegura troca de Roi
    gvars.l_itensToTrack[0].curRoi = 2
def pedal2():
    print("--- pedal 2 --------------------")
	# ME: encontra (R3); luz 1; ativa sobreposição (R4)
    gvars.l_itensToTrack[1].curRoi = 0
    gvars.l_itensToTrack[1].isActive = True
    gvars.l_itensToTrack[1].isProjecting = True
    gvars.l_itensToTrack[1].isBrightestForSure = True
    gvars.l_itensToTrack[1].allowOverlapRoiChange()
def pedal3():
    print("--- pedal 3 --------------------")
    # MD: ativa sobreposição (R6)
    gvars.l_itensToTrack[0].allowOverlapRoiChange()
	# ME: muda Roi (R5)
    gvars.l_itensToTrack[1].curRoi = 2
def pedal4():
    print("--- pedal 4 --------------------")
    # ME: muda Roi (R6)
    gvars.l_itensToTrack[1].curRoi = 3
def pedal5():
    print("--- pedal 5 --------------------")
    # MD: muda Roi (R7)
	# ME: muda Roi (R7)
def pedal6():
    print("--- pedal 6 --------------------")
    # MD: muda Roi (R8); ativa sobreposição (R10)
	# ME: muda Roi (R9)
def pedal7():
    print("--- pedal 7 --------------------")
    # MD: muda Roi (R11)
	# ME: muda Roi (R12)
def pedal8():
    print("--- pedal 8 --------------------")
    # MD: muda Roi (R13)
	# ME: muda Roi (R14)
def pedal9():
    print("--- pedal 9 --------------------")
    # MD: muda Roi (R15)
	# ME: luz 0