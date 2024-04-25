import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import os
def match_features(featuresA, featuresB, ratio):
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    rawMatches = bf.knnMatch(featuresA, featuresB, 2)
    matches = []
    for m, n in rawMatches:
        if m.distance < n.distance * ratio:
            matches.append(m)
    return matches

def getHomography(KeyPointsA, KeyPointsB, matches, reprojThresh):
    KeyPointsA = np.float32([kp.pt for kp in KeyPointsA])
    KeyPointsB = np.float32([kp.pt for kp in KeyPointsB])
    if len(matches) > 4:

        PointsA = np.float32([KeyPointsA[m.queryIdx] for m in matches])
        PointsB = np.float32([KeyPointsB[m.trainIdx] for m in matches])

        (H, status) = cv2.findHomography(PointsA, PointsB, cv2.RANSAC,
                                         reprojThresh)
        return (matches, H, status)
    else:
        return None
        
def merge(path1, path2, isShow=False):
    if isinstance(path2, str):
        imageA = cv2.imread(path2)
        imageA = cv2.cvtColor(imageA, cv2.COLOR_BGR2RGB)
    else:
        imageA = path2
    imageA_gray = cv2.cvtColor(imageA, cv2.COLOR_RGB2GRAY)

    if isinstance(path1, str):
        imageB = cv2.imread(path1)
        imageB = cv2.cvtColor(imageB, cv2.COLOR_BGR2RGB)
    else:
        imageB = path1
    imageB_gray = cv2.cvtColor(imageB, cv2.COLOR_RGB2GRAY)

    descriptor = cv2.SIFT_create()
    KeyPointsA, featuresA = descriptor.detectAndCompute(imageA_gray, None)
    KeyPointsB, featuresB = descriptor.detectAndCompute(imageB_gray, None)
    
    matches = match_features(featuresA, featuresB, ratio=0.75)
    matchCount = len(matches)
    M = getHomography(KeyPointsA, KeyPointsB, matches, reprojThresh=4)
    (matches, H, status) = M
    result = cv2.warpPerspective(imageA, H,
                                 ((imageA.shape[1] + imageB.shape[1]) * 2, (imageA.shape[0] + imageB.shape[0]) * 2))
    rows, cols = np.where(result[:, :, 0] != 0)
    min_row, max_row = min(rows), max(rows) + 1
    min_col, max_col = min(cols), max(cols) + 1
    result_img =result[min_row:max_row, min_col:max_col, :]
    if np.size(result_img) < np.size(imageA):
        KeyPointsA, KeyPointsB = KeyPointsB, KeyPointsA
        imageA, imageB = imageB, imageA
        matches = match_features(featuresB, featuresA, ratio=0.75)
        matchCount = len(matches)
        M = getHomography(KeyPointsA, KeyPointsB, matches, reprojThresh=4)
        (matches, H, status) = M
        result = cv2.warpPerspective(imageA, H,
                                     ((imageA.shape[1] + imageB.shape[1]) * 2, (imageA.shape[0] + imageB.shape[0]) * 2))
    result[0:imageB.shape[0], 0:imageB.shape[1]] = np.maximum(imageB, result[0:imageB.shape[0], 0:imageB.shape[1]])
    rows, cols = np.where(result[:, :, 0] != 0)
    min_row, max_row = min(rows), max(rows) + 1
    min_col, max_col = min(cols), max(cols) + 1
    result =result[min_row:max_row, min_col:max_col, :]
    return result, matchCount

def merge_all(img_paths, isShow=False):
    l = len(img_paths)
    pic_num = 1 
    ismerge = [0 for i in range(l - 1)] 
    nowPic = img_paths[0]
    img_paths = img_paths[1:]
    for j in range(l - 1):
        isHas = False  
        matchCountList = [] 
        resultList = [] 
        indexList = [] 
        for i in range(l - 1):
            if (ismerge[i] == 1):
                continue
            result, matchCount = merge(nowPic, img_paths[i])
            if not result is None:
                matchCountList.append(matchCount)
                resultList.append(result)
                indexList.append(i)
                isHas = True
        if not isHas: 
            if pic_num==1:
                return None
            else:
                return nowPic
        else:
            index = matchCountList.index(max(matchCountList))
            nowPic = resultList[index]
            pic_num+=1
            ismerge[indexList[index]] = 1
           
    return nowPic
    
if __name__ == "__main__":
    img_dir = 'C:/Users/dell/Desktop/pinjie'

    img_paths = []
    for img_name in os.listdir(img_dir):
        if "DS" not in img_name:
            img_paths.append(os.path.join(img_dir, img_name))

    if len(img_paths) > 2:
        result = merge_all(img_paths, isShow=False)

    else:
        result, _ = merge(img_paths[0], img_paths[1], isShow=False)

    if not result is None:
        cv2.imshow("result", result[:, :, [2, 1, 0]])
        cv2.imwrite(os.path.join(img_dir, 'merged_image.jpg'), result[:, :, [2, 1, 0]])
        plt.show()
        cv2.waitKey(0)
    exit()
