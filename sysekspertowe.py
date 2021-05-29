import numpy as np
import cv2
from scipy.ndimage import label
import random
from math import sqrt

def load_image(path_img):
    return cv2.imread(path_img)

def bgr2hsv(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

def setRangeColor(hsv, lower_color, upper_color):
    return cv2.inRange(hsv, lower_color, upper_color)

def contours_img(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def filter_contours_img(contours, img_draw, color_bbox):
    count = 0
    for c in contours:
        rect = cv2.boundingRect(c)
        x,y,w,h = rect
        area = w * h

        if area > 10000:
            count = count + 1
            cv2.rectangle(img_draw, (x, y), (x+w, y+h), color_bbox, 5)
    return img_draw, count

def draw_text_on_image(img_draw, count_parts,r,g,y,o,w):
    cv2.rectangle(img_draw, (0, 0), (500, 120), (0,0,0), -1)
    cv2.putText(img_draw,'liczba zelkow : ' + str(count_parts),
        (10,50),                  # bottomLeftCornerOfText
        cv2.FONT_HERSHEY_SIMPLEX, # font
        1.5,                      # fontScale
        (0,255,255),            # fontColor
        2)                        # lineType

    cv2.putText(img_draw,'czerwone:' + str(r)+'  zielone:' + str(g)+'  zolte:' + str(y)+'  pomaranczowe:' + str(o)+' biale:' + str(w),
        (10,100),                  # bottomLeftCornerOfText
        cv2.FONT_HERSHEY_SIMPLEX, # font
        1.5,                      # fontScale
        (0,255,255),            # fontColor
        2)                        # lineType

    return img_draw

def segment_on_dt(img):
        dt = cv2.distanceTransform(img, 2, 3)  # L2 norm, 3x3 mask
        dt = ((dt - dt.min()) / (dt.max() - dt.min()) * 255).astype(np.uint8)
        dt = cv2.threshold(dt, 100, 255, cv2.THRESH_BINARY)[1]
        lbl, ncc = label(dt)

        lbl[img == 0] = lbl.max() + 1
        lbl = lbl.astype(np.int32)
        cv2.watershed(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), lbl)
        lbl[lbl == -1] = 0
        return lbl
COLORS = (
        (120,'red'),
        (87,'green'),
        (93,'yellow'),
        (104,'orange'),
        (96, 'white'),

)
def closest_color(r):

    color_diffs = []
    for color in COLORS:
        cr, str = color
        color_diff = sqrt(abs(r - cr)**2)
        color_diffs.append((color_diff, color))
    return min(color_diffs)[1]

def main():
    for i in range(1,11):
        count_parts = 0
        path_img = 'images/'+str(i)+'.jpg'
        img = load_image(path_img)
        img = cv2.resize(img, None,fx=0.5,fy=0.5)
        hsv = bgr2hsv(img)



        lower = np.array([0,52,0])
        upper = np.array([255,255,255])
        mask = setRangeColor(hsv, lower, upper)
        kernel = np.ones((3, 3), np.uint8)
        dilate = cv2.dilate(mask, kernel, iterations=2)
        ws_result = segment_on_dt(dilate)
        #cv2.imshow('a', cv2.resize(dilate, None, fx=0.5, fy=0.5))
        #cv2.waitKey(0)

        height, width = ws_result.shape
        ws_color = np.zeros((height, width, 3), dtype=np.uint8)
        lbl, ncc = label(ws_result)
        for l in range(1, ncc + 1):
            a, b = np.nonzero(lbl == l)
            if dilate[a[0], b[0]] == 0:  # Do not color background.
                continue
            rgb = [random.randint(0, 255) for _ in range(3)]
            ws_color[lbl == l] = tuple(rgb)
            #cv2.imshow('aaaaaa', cv2.resize(ws_color, None,fx=0.5,fy=0.5))

        wsbin = np.zeros((height, width), dtype=np.uint8)
        wsbin[cv2.cvtColor(ws_color, cv2.COLOR_BGR2GRAY) != 0] = 255
        listnum=[]
        ws_bincolor = cv2.cvtColor(255 - wsbin, cv2.COLOR_GRAY2BGR)
        lbl, ncc = label(wsbin)
        for l in range(1, ncc + 1):
            yx = np.dstack(np.nonzero(lbl == l)).astype(np.int64)
            xy = np.roll(np.swapaxes(yx, 0, 1), 1, 2)
            if len(xy) < 100:  # Too small.
                continue
            ellipse = cv2.fitEllipse(xy)
            center, axes, angle = ellipse
            rect_area = axes[0] * axes[1]

            if rect_area > 5000 and rect_area < 35000 :
                if 0.8 < rect_area / float(len(xy)) < 1.2:
                        count_parts += 1
                        rect = np.round(np.float64(
                            cv2.boxPoints(ellipse))).astype(np.int64)
                        color = [random.randint(60, 255) for _ in range(3)]
                        cv2.drawContours(ws_bincolor, [rect], 0, color, 2)
                        x, y, w, h = cv2.boundingRect(rect)  # offsets - with this you get 'mask'4
                        cuted = img[int(y + h / 3):int(y + h / 2), int(x + w / 3):int(x + w / 2)]
                        cv2.rectangle(img, (x, y), (x + w, y + h), cv2.mean(cuted), 8)
                        #cv2.imshow('', cuted)
                        color_mean = np.round(cv2.mean(cuted)).astype(np.int64)
                        cuted = cv2.cvtColor(cuted, cv2.COLOR_RGB2HSV)
                        color_meanhsv = np.round(cv2.mean(cuted)).astype(np.int64)
                        #print('Average color (bgr): ', color_mean)
                        #print('Average color (hsv): ', color_meanhsv)
                        print(closest_color((color_meanhsv[0]))[1])
                        listnum.append(closest_color((color_meanhsv[0]))[1])

                        #cv2.waitKey(0)

        img = draw_text_on_image(img,count_parts,listnum.count('red'),listnum.count('green'),listnum.count('yellow'),listnum.count('orange'),listnum.count('white'))
        cv2.imshow('a', cv2.resize(img, None, fx=0.5, fy=0.5))
        cv2.waitKey( 0 )

if __name__ == '__main__':
    main()






