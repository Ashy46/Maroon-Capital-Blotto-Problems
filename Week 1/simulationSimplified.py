def simulate():
    bestEv = 0
    bestHand = []
    for x1 in range(21):
        for x2 in range(21 - x1):
            for x3 in range(21 - x1 -x2):
                for x4 in range(21 - x1 - x2 - x3):
                    for x5 in range(21 - x1 - x2 - x3 - x4):
                        for x6 in range(21 - x1 - x2 - x3 - x4 - x5):
                            for x7 in range(21 - x1 - x2 - x3 - x4 - x5 - x6):
                                for x8 in range(21 - x1 - x2 - x3 - x4 - x5  -x6 - x7):
                                    for x9 in range(21 - x1 - x2 - x3 - x4 - x5 - x6 -x7 - x8):
                                        x10 = 20 - x1 - x2 - x3 - x4 - x5 - x6 -x7 - x8 - x9
                                        mySetUp = [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10]
                                        temp = check_each(mySetUp)
                                        
                                        if temp > bestEv:
                                            bestHand = mySetUp
                                            bestEv = temp
                                            bestHand.append(bestEv)
                                            print(bestHand)
    return bestHand

    
def check_each(mySetUp):
    worstEv = 100000
    for x1 in range(21):
        for x2 in range(21 - x1):
            for x3 in range(21 - x1 -x2):
                for x4 in range(21 - x1 - x2 - x3):
                    for x5 in range(21 - x1 - x2 - x3 - x4):
                        for x6 in range(21 - x1 - x2 - x3 - x4 - x5):
                            for x7 in range(21 - x1 - x2 - x3 - x4 - x5 - x6):
                                for x8 in range(21 - x1 - x2 - x3 - x4 - x5  -x6 - x7):
                                    for x9 in range(21 - x1 - x2 - x3 - x4 - x5 - x6 -x7 - x8):
                                            x10 = 20 - x1 - x2 - x3 - x4 - x5 - x6 -x7 - x8 - x9
                                            temp  = 0
                                            if mySetUp[0] >= x1:
                                                if mySetUp[0] == x1:
                                                    temp += 1/2
                                                else:
                                                    temp += 1
                                            if mySetUp[1] >= x2:
                                                if mySetUp[1] == x2:
                                                    temp += 2/2
                                                else:
                                                    temp += 2
                                            if mySetUp[2] >= x3:
                                                if mySetUp[2] == x3:
                                                    temp += 3/2
                                                else:
                                                    temp += 3
                                            if mySetUp[3] >= x4:
                                                if mySetUp[3] == x4:
                                                    temp += 4/2
                                                else:
                                                    temp += 4
                                            if mySetUp[4] >= x5:
                                                if mySetUp[0] == x5:
                                                    temp += 5/2
                                                else:
                                                    temp += 5
                                            if mySetUp[5] >= x6:
                                                if mySetUp[5] == x6:
                                                    temp += 6/2
                                                else:
                                                    temp += 6
                                            if mySetUp[6] >= x7:
                                                if mySetUp[6] == x7:
                                                    temp += 7/2
                                                else:
                                                    temp += 7
                                            if mySetUp[7] >= x8:
                                                if mySetUp[7] == x8:
                                                    temp += 8/2
                                                else:
                                                    temp += 8
                                            if mySetUp[8] >= x9:
                                                if mySetUp[8] == x9:
                                                    temp += 9/2
                                                else:
                                                    temp += 9
                                            if mySetUp[9] >= x10:
                                                if mySetUp[9] == x10:
                                                    temp += 10/2
                                                else:
                                                    temp += 10
                                            
                                            if temp < worstEv:
                                                worstEv = temp
    return worstEv

print(simulate())