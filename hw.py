import cv2 
import glob
import os 
from matplotlib import pyplot as plt
import numpy as np

# constants 
ROW = 60
COL = 89
IMG_COUNT = 40
BLACK = 71
BLACK_SYMM = 82
# Color: this is the bin partition that will determine the accuracy of color distinguishment
BIN_PARTITION = [2, 6, 4]
BIN_2 = 64
STANDARD_SIMPLEX = [0.39, 0.21, 0.23, 0.17]
MY_STANDARD_SIMPLEX = [0.39, 0.21, 0.23, 0.17]

'''This is the function that calculates the normalized L1 distance between two images'''
def calculateL1Norm(hist1, hist2):
    re = 0
    for B in np.arange(hist1.shape[0]):
        for G in np.arange(hist1.shape[1]):
            for R in np.arange(hist1.shape[2]):
                re += abs(hist1[B, G, R] - hist2[B, G, R])/(2 * ROW * COL) 
    return re 


'''This is the function that processes Crowd.txt and 
it returns an array of array so Crowd(q, t1) can be accessed by Crowd[q-1][t1-1]'''
def processCrowd(text):
    crowd = open(text, 'r')
    lines = crowd.readlines()
    li = []
    for i in range (0, len(lines)):
        ele = lines[i].strip().split(' ')
        c = ele.count('')
        for i in range(c):
            ele.remove('')
        li.append(ele)
    return li


'''This is the function that processes myPreferences.txt'''
def processSelfPref():
    crowd = open('myPreferences.txt', 'r')
    lines = crowd.readlines()
    d = {}
    for i in range (0, len(lines)):
        ele = lines[i].strip().split(' ')
        c = ele.count('')
        for i in range(c):
            ele.remove('')
        d[ele[0]] = ele[1:]
    return d


'''Step 1.3 This is the function that writes the result of the system vs. crowd preferences test to an html file'''
def writeToCrowdScore(num, score, matched_imgs, iteration, crowd, score_sum, step):
    with open('system_vs_crowd_pref'+str(step)+'.html', 'a') as the_file:
        the_file.write('''<span style="font-weight: bolder">'''+num+'''</span>\n''')
        the_file.write('''<img src = "images/i''' + num + '''.jpg" width = "80px" height = "80px" style="padding-right: 20">\n''')
        for img in matched_imgs: 
            the_file.write('''<span style="font-weight: bolder">'''+ img + ''', </span>\n''')
            the_file.write('''<span> crowd('''+ str(int(num)) + ''', ''' + img +'''): ''' + crowd[int(num)-1][int(img)-1] + '''</span>''')
            the_file.write('''<img class="col" src = "images/i''' + img + '''.jpg" width = "80px" height = "80px" style="padding-right: 20">\n''')
        the_file.write('''<span> Score: ''' + str(score) + '''</span>''')
        the_file.write('''<br>\n''')
        if iteration == IMG_COUNT - 1 and step != 5 and step != 6 or iteration == IMG_COUNT and step == 5 or iteration == IMG_COUNT and step == 6:
            the_file.write('''<br>\n''')
            the_file.write('''<span style="font-weight: bolder">Overall Score: '''+ str(score_sum) + '''</span>''')
            the_file.close()


'''Step 1.3 This is the function that calculates the result of system vs. personal preferences'''
def calcSelfPref(myPref, match):
    num = 0
    for q in myPref:
        myPrefMatches = set(myPref[q])
        res = set(match[q])
        num += len(myPrefMatches.intersection(res))
    return num


'''Step 2.2: this is the function that converts an image to grayscale by using I=(R+G+B)/3'''
def convertToGrayScale(img):
    grayscale = []
    for r in range (ROW):
        for c in range (COL):
            pixel = img[r][c]
            new_pixel = round((int(pixel[0])+int(pixel[1])+int(pixel[2]))/3)
            grayscale.append(new_pixel)
    gray = np.reshape(grayscale, (ROW, COL))
    return gray


'''Step 2.2: This is the function that calculates the Laplacian image of an image'''
def calcLap(gray):
    laplacian = []
    for r in range (ROW):
        for c in range (COL):
            val = 0
            if not (r == 0 or c == 0 or r == ROW-1 or c == COL-1):
                val = 8 * gray[r][c] - gray[r-1][c] - gray[r+1][c] - gray[r][c-1]
                val = val - gray[r][c+1] - gray[r-1][c-1] - gray[r-1][c+1]
                val = val - gray[r+1][c-1] - gray[r+1][c+1]
            elif r == 0 and c != 0 and c != COL-1: 
                val = 8 * gray[r][c] - gray[r+1][c] - gray[r][c-1] - gray[r][c+1] - gray[r+1][c-1] - gray[r+1][c+1]
            elif r == ROW-1 and c != 0 and c != COL-1:
                val = 8 * gray[r][c] - gray[r-1][c] - gray[r][c-1] - gray[r][c+1] - gray[r-1][c-1] - gray[r-1][c+1]
            elif c == 0 and r != 0 and r != ROW-1:
                val = 8 * gray[r][c] - gray[r-1][c] - gray[r+1][c] - gray[r][c+1] - gray[r-1][c+1] - gray[r+1][c+1]
            elif c == COL-1 and r != 0 and r != ROW-1:
                val = 8 * gray[r][c] - gray[r-1][c] - gray[r+1][c] - gray[r][c-1] - gray[r-1][c-1] - gray[r+1][c-1] 
            elif c == 0 and r == 0:
                val = 8 * gray[r][c] - gray[r+1][c] - gray[r][c+1] - gray[r+1][c+1]
            elif r == 0 and c == COL-1:
                val = 8 * gray[r][c] - gray[r+1][c] - gray[r][c-1] - gray[r+1][c-1]
            elif r == ROW-1 and c == 0:
                val = 8 * gray[r][c] - gray[r-1][c] - gray[r][c+1] - gray[r-1][c+1]
            elif c == COL-1 and r == ROW-1:
                val = 8 * gray[r][c] - gray[r-1][c] - gray[r][c-1] - gray[r-1][c-1]
            else:
                print('ERROR')
            laplacian.append(abs(val))
    return (laplacian, np.reshape(laplacian, (ROW, COL)))


'''Step 2.2: This is the function that normalizes the textural distance'''
def textureDistNormalize(hist1, hist2):
    re = 0
    for ele1, ele2 in zip(hist1, hist2):
        re += abs(ele1-ele2)/(2*ROW*COL)
    return re


'''Step 3: This is the function that converts an 8-bit intensity image to 1 bit black and white'''
def convertToBlackWhite(img, thresh):
    for r in range (ROW):
        for c in range (COL):
            if img[r][c][0] > thresh and img[r][c][1] > thresh and img[r][c][2] > thresh:
                img[r][c]= (255, 255, 255)
            else:
                img[r][c] = (0, 0, 0)
    return img


'''Step 3: This is the function that calculates the “normalized overlap distance" between two images'''
def calcNormalizedOverlapDist(img1, img2):
    summ = 0
    for i in range (0, len(img1)):
        for j in range (0, len(img1[0])):
            if img1[i][j][0] != img2[i][j][0] or img1[i][j][1] != img2[i][j][1] or img1[i][j][2] != img2[i][j][2]:
                summ += 1
    summ = summ/(ROW * COL)
    return summ


'''Step 4: This is the function that folds the image in half and calculates its normalized symmetry'''
def calcNormalizedSymmetry(img):
    summ = 0
    for i in range (0, ROW):
        for j in range (0, COL//2):
            p1 = img[i][j]
            p2 = img[i][COL-j-1]
            if p1[0] != p2[0] or p1[1] != p2[1] or p1[2] != p2[2]:
                summ += 1
    summ = summ / (ROW * COL / 2)
    return summ


def color(li_img, crowd, myPref):
    # match is a dictionary that keeps track of the three closest images to a given image 
    match = {}
    # the score dictionary keeps track of the score from crowd.txt for each image
    score = {}
    # score of the system given Crowd.txt
    score_sum = 0
    # color distance
    color_dist = {}
    # compare the images to each other 
    for i in range (0, IMG_COUNT):
        name = li_img[i]
        num = name[-6:-4]
        img1 = cv2.imread(name)
        
        # this is the list that keeps track of all the l1 distance between img1 and other 39 images
        # each entry is in the format of (l1_distance, number_of_other_image)
        l1_li = []
        
        for j in range (0, IMG_COUNT):
            if i == j: 
                continue
            name2 = li_img[j]
            num2 = name2[-6:-4]
            img2 = cv2.imread(li_img[j])
            # calculate histogram 
            ppm_hist1 = cv2.calcHist([img1],[0, 1, 2], None, BIN_PARTITION, [0, 256, 0, 256, 0, 256])
            ppm_hist2 = cv2.calcHist([img2],[0, 1, 2], None, BIN_PARTITION, [0, 256, 0, 256, 0, 256])
            # calculate the L1 norm between each image 
            result = calculateL1Norm(ppm_hist1, ppm_hist2)
            l1_li.append((result, num2))
        color_dist[int(num)] = l1_li
        l1_li.sort()
        # extract the three that has the smallest normalized L1 distance 
        matched_imgs = (l1_li[0][1], l1_li[1][1], l1_li[2][1])
        match[num] = matched_imgs 
        # calculate the score given Crowd.txt 
        s = int(crowd[int(num)-1][int(matched_imgs[0])-1]) + int(crowd[int(num)-1][int(matched_imgs[1])-1]) + int(crowd[int(num)-1][int(matched_imgs[2])-1])
        score[num] = s
        score_sum += s 
        writeToCrowdScore(num, s, matched_imgs, i, crowd, score_sum, 1)
    selfPrefResColor = calcSelfPref(myPref, match)

    # This is used to display results on terminal 
    print('Matching with Color: ', match)
    print('Total score of Color: ', score_sum)
    print('Self Satisfaction Score -- Color: ', selfPrefResColor)
    return (match, score_sum, selfPrefResColor, color_dist)


def texture(crowd, myPref):
    #convert images to grayscale
    dic = {}
    score_texture = 0
    score_step2 = {}
    texture_dist = {}
    for i in range (0, IMG_COUNT):
        li_i = []
        name = li_img[i]
        num1 = name[-6:-4]
        img = cv2.imread(name, cv2.IMREAD_UNCHANGED)
        grayscale = convertToGrayScale(img).astype(np.uint8)
        lap_1_img = cv2.Laplacian(grayscale,cv2.CV_8U,ksize=3)
        lap_img1 = cv2.convertScaleAbs(lap_1_img)
        cv2.imwrite('images/'+str(num1)+'lap.jpg', lap_img1)
        hist1 = cv2.calcHist([lap_img1], [0], None, [BIN_2], [0, 256])
        for j in range (0, IMG_COUNT):
            if i == j:
                continue
            name2 = li_img[j]
            num2 = name2[-6:-4]
            img2 = cv2.imread(name2)
            # covert to grayscale image 
            grayscale_2 = convertToGrayScale(img2).astype(np.uint8)
            # calculate Laplacian image 
            lap_2_img = cv2.Laplacian(grayscale_2,cv2.CV_8U,ksize=3)
            lap_img2 = cv2.convertScaleAbs(lap_2_img)
            hist2 = cv2.calcHist([lap_img2], [0], None, [BIN_2], [0, 256])
            result = textureDistNormalize(hist1, hist2)
            li_i.append((result, num2))
        texture_dist[int(num1)] = li_i
        li_i.sort()
        matched_imgs = (li_i[0][1], li_i[1][1], li_i[2][1])
        dic[num1] = matched_imgs
        s = int(crowd[int(num1)-1][int(matched_imgs[0])-1]) + int(crowd[int(num1)-1][int(matched_imgs[1])-1]) + int(crowd[int(num1)-1][int(matched_imgs[2])-1])
        score_step2[num1] = s
        score_texture += s 
        writeToCrowdScore(num1, s, matched_imgs, i, crowd, score_texture, 2)
    selfPrefResTexture = calcSelfPref(myPref, dic)
    # This is used to display results on terminal 
    print('Matching with Texture: ', dic)
    print('Total score of Texture: ', score_texture)
    print('Self Satisfaction Score -- Texture: ', selfPrefResTexture)
    return (dic, score_texture, selfPrefResTexture, texture_dist)


def shape(crowd, myPref):
    shape_match = {}
    score_shape = 0
    score_step3 = {}
    shape_dist = {}
    # convert laplacian images to binary 
    bw_li = []
    for lap in glob.glob(os.path.join("images", '*gray.jpg')): 
        img = cv2.imread(lap)
        black_and_white = convertToBlackWhite(img, BLACK)
        cv2.imwrite(lap[:-8]+'shape.jpg', black_and_white)
        bw_li.append((lap[-10:-8], black_and_white))
    bw_li.sort()
    for i in range (0, IMG_COUNT):
        num1 = bw_li[i][0]
        img1 = bw_li[i][1]
        li = []
        for j in range (0, IMG_COUNT):
            if i == j:
                continue
            num2 = bw_li[j][0]
            img2 = bw_li[j][1]
            re = calcNormalizedOverlapDist(img1, img2)
            li.append((re, num2))
        shape_dist[int(num1)] = li
        li.sort()
        matched_imgs = (li[0][1], li[1][1], li[2][1])
        shape_match[num1] = matched_imgs
        s = int(crowd[int(num1)-1][int(matched_imgs[0])-1]) + int(crowd[int(num1)-1][int(matched_imgs[1])-1]) + int(crowd[int(num1)-1][int(matched_imgs[2])-1])
        score_step3[num1] = s
        score_shape += s 
        writeToCrowdScore(num1, s, matched_imgs, i, crowd, score_shape, 3)
    selfPrefResShape = calcSelfPref(myPref, shape_match)
    print('Matching with Shape: ', shape_match)
    print('Total score of Shape: ', score_shape)
    print('Self Satisfaction Score -- Shape: ', selfPrefResShape)
    return (shape_match, score_shape, selfPrefResShape, shape_dist)


def symmetry(crowd, myPref):
    symm_li = []
    score_symmetry = 0
    score_step4 = {}
    symmetry_match = {}
    symmetry_dist = {}
    bw_li = []
    for gray in glob.glob(os.path.join("images", '*gray.jpg')): 
        img = cv2.imread(gray)
        black_and_white = convertToBlackWhite(img, BLACK_SYMM)
        cv2.imwrite(gray[:-8]+'symme.jpg', black_and_white)
        bw_li.append((gray[-10:-8], black_and_white))
    bw_li.sort()
    for i in range (0, IMG_COUNT):
        num1 = bw_li[i][0]
        img1 = bw_li[i][1]
        re = calcNormalizedSymmetry(img1)
        symm_li.append((re, num1))
    for i in range (0, len(symm_li)):
        val, num1 = symm_li[i]
        li = []
        for j in range (0, len(symm_li)):
            if i == j:
                continue
            val2, num2 = symm_li[j]
            li.append((abs(val2-val), num2))
        symmetry_dist[int(num1)] = li
        li.sort()
        matched_imgs = (li[0][1], li[1][1], li[2][1])
        symmetry_match[num1] = matched_imgs
        s = int(crowd[int(num1)-1][int(matched_imgs[0])-1]) + int(crowd[int(num1)-1][int(matched_imgs[1])-1]) + int(crowd[int(num1)-1][int(matched_imgs[2])-1])
        score_step4[num1] = s
        score_symmetry += s 
        writeToCrowdScore(num1, s, matched_imgs, i, crowd, score_symmetry, 4)
    selfPrefResSymmetry = calcSelfPref(myPref, symmetry_match)
    print('Matching with Symmetry: ', symmetry_match)
    print('Total score of Symmetry: ', score_symmetry)
    print('Self Satisfaction Score -- Symmetry: ', selfPrefResSymmetry)
    return (symmetry_match, score_symmetry, selfPrefResSymmetry, symmetry_dist)


def overall(crowd, myPref):
    overall_score = {}
    score_step5 = {}
    overall = 0
    for i in range (1, IMG_COUNT+1):
        c_dist = color_dist[i]
        t_dist = texture_dist[i]
        sh_dist = shape_dist[i]
        sy_dist = symmetry_dist[i]
        overall_score[i] = 0
        li = []
        score = {1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0, 13:0, 14:0, 15:0, 16:0, 17:0, 18:0, 19:0, 20:0, 21:0, 22:0, 23:0, 24:0, 25:0, 26:0, 27:0, 28:0, 29:0, 30:0, 31:0, 32:0, 33:0, 34:0, 35:0, 36:0, 37:0, 38:0, 39:0, 40:0}
        for c, t, sh, sy in zip(c_dist, t_dist, sh_dist, sy_dist):
            val_c, ele_c = c[0], c[1]
            val_t, ele_t = t[0], t[1]
            val_sh, ele_sh = sh[0], sh[1]
            val_sy, ele_sy = sy[0], sy[1]
            score[int(ele_c)] += val_c * STANDARD_SIMPLEX[0]
            score[int(ele_t)] += val_t * STANDARD_SIMPLEX[1]
            score[int(ele_sh)] += val_sh * STANDARD_SIMPLEX[2]
            score[int(ele_sy)] += val_sy * STANDARD_SIMPLEX[3]
        for ele in score:
            new_ele = ele
            if new_ele < 10: 
                new_ele = str(0) + str(ele)
            li.append((score[ele], str(new_ele)))
        li.sort()
        matched_imgs = (li[1][1], li[2][1], li[3][1])
        if i < 10: 
            i = str(0) + str(i)
        overall_score[str(i)] = matched_imgs
        s = int(crowd[int(i)-1][int(matched_imgs[0])-1]) + int(crowd[int(i)-1][int(matched_imgs[1])-1]) + int(crowd[int(i)-1][int(matched_imgs[2])-1])
        score_step5[i] = s
        overall += s 
        writeToCrowdScore(str(i), s, matched_imgs, i, crowd, overall, 5)
    selfPrefResOverall = calcSelfPref(myPref, overall_score)
    print('Total score: ', overall)
    print('Self Satisfaction Score -- Overall: ', selfPrefResOverall)


'''Step 6: This is the function that creates my personal Crowd.txt'''
def writeToMyCrowd(myPref):
    pref_mat = np.zeros((40,40))
    with open("myCrowd.txt", "a") as f:
        for i in myPref:
            photo_id = int(i)-1
            pref_mat[photo_id, int(myPref[i][0])-1] = 3
            pref_mat[photo_id, int(myPref[i][1])-1] = 2
            pref_mat[photo_id, int(myPref[i][2])-1] = 1
    np.savetxt("myCrowd.txt", pref_mat.astype(int), fmt='%i')
    
    
def myOverall(mycrowd, myPref):
    overall_score = {}
    score_step6 = {}
    my_overall_re = 0
    for i in range (1, IMG_COUNT+1):
        c_dist = color_dist[i]
        t_dist = texture_dist[i]
        sh_dist = shape_dist[i]
        sy_dist = symmetry_dist[i]
        overall_score[i] = 0
        li = []
        score = {1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0, 13:0, 14:0, 15:0, 16:0, 17:0, 18:0, 19:0, 20:0, 21:0, 22:0, 23:0, 24:0, 25:0, 26:0, 27:0, 28:0, 29:0, 30:0, 31:0, 32:0, 33:0, 34:0, 35:0, 36:0, 37:0, 38:0, 39:0, 40:0}
        for c, t, sh, sy in zip(c_dist, t_dist, sh_dist, sy_dist):
            val_c, ele_c = c[0], c[1]
            val_t, ele_t = t[0], t[1]
            val_sh, ele_sh = sh[0], sh[1]
            val_sy, ele_sy = sy[0], sy[1]
            score[int(ele_c)] += val_c * MY_STANDARD_SIMPLEX[0]
            score[int(ele_t)] += val_t * MY_STANDARD_SIMPLEX[1]
            score[int(ele_sh)] += val_sh * MY_STANDARD_SIMPLEX[2]
            score[int(ele_sy)] += val_sy * MY_STANDARD_SIMPLEX[3]
        for ele in score:
            new_ele = ele
            if new_ele < 10: 
                new_ele = str(0) + str(ele)
            li.append((score[ele], str(new_ele)))
        li.sort()
        matched_imgs = (li[1][1], li[2][1], li[3][1])
        if i < 10: 
            i = str(0) + str(i)
        overall_score[str(i)] = matched_imgs
        s = int(mycrowd[int(i)-1][int(matched_imgs[0])-1]) + int(mycrowd[int(i)-1][int(matched_imgs[1])-1]) + int(mycrowd[int(i)-1][int(matched_imgs[2])-1])
        score_step6[i] = s
        my_overall_re += s 
        writeToCrowdScore(str(i), s, matched_imgs, i, mycrowd, my_overall_re, 6)
    selfPrefResOverall = calcSelfPref(myPref, overall_score)
    print('Total score: ', my_overall_re)
    print('Self Satisfaction Score -- Overall: ', selfPrefResOverall)


'''----------MAIN FUNCTION----------'''
li_img = []
if os.path.exists("system_vs_crowd_pref1.html"):
    os.remove("system_vs_crowd_pref1.html")
if os.path.exists("system_vs_crowd_pref2.html"):
    os.remove("system_vs_crowd_pref2.html")
if os.path.exists("system_vs_crowd_pref3.html"):
    os.remove("system_vs_crowd_pref3.html")
if os.path.exists("system_vs_crowd_pref4.html"):
    os.remove("system_vs_crowd_pref4.html")
if os.path.exists("system_vs_crowd_pref5.html"):
    os.remove("system_vs_crowd_pref5.html")
if os.path.exists("system_vs_crowd_pref6.html"):
    os.remove("system_vs_crowd_pref6.html")
for ppm_name in glob.glob(os.path.join("images", '*.ppm')):
    # create a list of all loaded images 
    li_img.append(ppm_name)
# sort the list so that the images are read from 1 to 40
li_img = sorted(li_img, key=lambda n: int(n[-6:-4:]))
# read and process Crowd.txt so Crowd(q, t1) can be accessed by Crowd[q-1][t1-1]
crowd = processCrowd('Crowd.txt')
# read the process myPreferences.txt and return a dictionary with key being q and value being the three image names 
myPref = processSelfPref()
mycrowd = processCrowd('myCrowd.txt')

'''Step 1-4'''
color_match, score_color, self_color_score, color_dist = color(li_img, crowd, myPref)
texture_match, score_texture, self_texture_score, texture_dist = texture(crowd, myPref)
shape_match, score_shape, self_shape_score, shape_dist = shape(crowd, myPref)
symmetry_match, score_symmetry, self_symmetry_score, symmetry_dist = symmetry(crowd, myPref)

'''Step 5: Overall'''
overall(crowd, myPref)

'''Step 6: my Overall'''
writeToMyCrowd(myPref)
myOverall(mycrowd, myPref)