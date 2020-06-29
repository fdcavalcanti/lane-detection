import cv2
import numpy as np
import matplotlib.pyplot as plt

#%%
# How it works:
# Preprocess frame: get ROI in BW and Canny
# Get lines: 

# Constants
# Resize values
new_H = 350
new_W = 500

# BGR 2 GRAY conversion
low_thresh_yellow = np.array([10, 38, 115])
high_thresh_yellow = np.array([35, 204, 255])
low_thresh_white = np.array([0, 0, 170])
high_thresh_white = np.array([180, 40, 200])

# ROI Values
roi_area = [np.array([(0,250), (150,170), (270,170), (450,350)])]
empty_canvas = np.zeros((new_H, new_W), dtype=np.uint8)
mask_roi = np.zeros_like(empty_canvas, dtype=np.uint8)
cv2.fillPoly(mask_roi, roi_area, 255)

# Canny Values
canny_thresh_low = 70
canny_thresh_high = 3*canny_thresh_low

# Hough Values
minLineLength = 10
maxLineGap = 10
thresholdHough = 15

# Lines simplifying 
x_points = np.arange(0, new_W)
y_max_draw = 0.5


#%%
def get_main_lines(frame):
    lines = cv2.HoughLinesP(frame, 1, np.pi/180, thresholdHough, np.array([]), minLineLength, maxLineGap)
    left_lines = []
    right_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # m, b refers to y = m*x + b (line equation)
            m, b = np.polyfit([x1,x2], [y1,y2], 1)
            if m > 0:
                right_lines.append(line)
            else:
                left_lines.append(line)
        
    return np.array(left_lines), np.array(right_lines)

def find_left_and_outside_lines(lines):
    #https://stackoverflow.com/questions/1560492/how-to-tell-whether
    #-a-point-is-to-the-right-or-left-side-of-a-line
    # Calcula uma linha média para lines e separa as linhas que estão a direita e a equerda
    # desta linha média
    outside_lines, left_lines = [], []
    if len(lines) > 0:
        x1_mean = int(np.mean(lines[:,:,0]))
        y1_mean = int(np.mean(lines[:,:,1]))
        x2_mean = int(np.mean(lines[:,:,2]))
        y2_mean = int(np.mean(lines[:,:,3]))
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            pos1 = np.sign((x2_mean - x1_mean) * (y1 - y1_mean) - (y2_mean - y1_mean) * (x1 - x1_mean))
            pos2 = np.sign((x2_mean - x1_mean) * (y2 - y1_mean) - (y2_mean - y1_mean) * (x2 - x1_mean))
            #print("pos 1:" + str(pos1) + " pos 2: " + str(pos2))
            #print("  " + str(line[0]))
            if (pos1 > 0 and pos2 > 0):
                left_lines.append(line)
                #plt.plot([x1, x2],[y1, y2], c='cyan')
            elif (pos1 < 0 and pos2 < 0):
                outside_lines.append(line)
                #plt.plot([x1, x2],[y1, y2], c='magenta')
        
        #plt.plot([x1_mean, x2_mean],[y1_mean, y2_mean], c='orange')
    
    return np.array(outside_lines), np.array(left_lines)
                
def simplify_set_of_lines(lines):
    # Simplifica o conjunto de linhas em uma linha só
    # Define o Y e acha X para limitar o quanto da linha pode ser desenhada na tela
    if len(lines) > 0:
        x = np.concatenate((lines[:,:,0], lines[:,:,2])).flatten()
        y = np.concatenate((lines[:,:,1], lines[:,:,3])).flatten()
        m, b = np.polyfit(x, y, 1)
        y1 = int(new_H)
        y2 = int(new_H * y_max_draw)
        x1 = int((y1 - b) / m)
        x2 = int((y2 - b) / m)
        
        return np.array([x1, y1, x2, y2])
    
    else:
        print('def simplify_set_of_lines: not enough points ')
        return np.array([0,0,0,0])
    
#%%
def convert_to_gray(frame_bgr):
    # Isolates yellow and white from the HSV frame and converts it to gray. 
    # HSV range in OpenCV: H[0,179] S[0,255] V[0,255]
    # http://colorizer.org/ HSV Ranges: H[0,360] S[0,100] V[0,100]
    frame_hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    yellow_mask = cv2.inRange(frame_hsv, low_thresh_yellow, high_thresh_yellow)
    white_mask = cv2.inRange(frame_hsv, low_thresh_white, high_thresh_white)
    full_mask = cv2.bitwise_or(yellow_mask, white_mask)
    frame_masked = cv2.bitwise_and(frame_hsv, frame_hsv, mask=full_mask)
    
    frame_masked_bgr = cv2.cvtColor(frame_masked, cv2.COLOR_HSV2BGR)
    frame_masked_gray = cv2.cvtColor(frame_masked_bgr, cv2.COLOR_BGR2GRAY)
    
    return frame_masked_gray

def preprocess_frame(frame):
    frame_res = cv2.resize(frame, (new_W, new_H))
    frame_gray = convert_to_gray(frame_res)
    frame_roi = cv2.bitwise_and(frame_gray, mask_roi)
    frame_canny = cv2.Canny(frame_roi, canny_thresh_low, canny_thresh_high)
    
    return frame_res, frame_canny

#%%
def draw_lines(outside, left, right, simplify):
    canvas = np.zeros((new_H, new_W, 3))
    
    if simplify == 1:
        if len(outside) > 0:
            x1,y1,x2,y2 = simplify_set_of_lines(outside)
            cv2.line(canvas, (x1,y1), (x2,y2), (255,0,0), 2)
        
        if len(left) > 0:
            x1,y1,x2,y2 = simplify_set_of_lines(left)
            cv2.line(canvas, (x1,y1), (x2,y2), (0,255,0), 2)
        
        if len(right) > 0:
            x1,y1,x2,y2 = simplify_set_of_lines(right)
            cv2.line(canvas, (x1,y1), (x2,y2), (0,0,255), 2)
    
    else:
        if len(outside) > 0:
            for line in outside:
                x1,y1,x2,y2 = line[0]
                cv2.line(canvas, (x1,y1), (x2,y2), (255,0,0), 1)
        
        if len(left) > 0:
            for line in left:
                x1,y1,x2,y2 = line[0]
                cv2.line(canvas, (x1,y1), (x2,y2), (0,255,0), 1)
        
        if len(right) > 0:
            for line in right:
                x1,y1,x2,y2 = line[0]
                cv2.line(canvas, (x1,y1), (x2,y2), (0,0,255), 1)
    
    return canvas.astype(np.uint8)

#%%
frame_orig = cv2.imread('test_image.jpg')
H, W = frame_orig.shape[0], frame_orig.shape[1]
frame_res, frame_canny = preprocess_frame(frame_orig)

other_lines, right_lines = get_main_lines(frame_canny)
outside_lines, left_lines = find_left_and_outside_lines(other_lines)

canvas = draw_lines(outside_lines, left_lines, right_lines, 1)
canvas_overlay = cv2.addWeighted(frame_res, 0.5, canvas, 1, 1)

plt.imshow(canvas_overlay)

'''
for line in right_lines:
    x1,y1,x2,y2 = line[0]
    plt.plot([x1, x2],[y1, y2], c='b')
plt.gca().invert_yaxis()
plt.xlim((0,new_W))
plt.ylim((new_H, 0))

x1,y1,x2,y2 = simplify_set_of_lines(left_lines)
plt.plot([x1, x2],[y1, y2])
x1,y1,x2,y2 = simplify_set_of_lines(outside_lines)
plt.plot([x1, x2],[y1, y2])

canvas = np.zeros_like(frame_res)
cv2.line(canvas, (x1,y1), (x2,y2), (0,255,0), 2)

'''
#position = sign((Bx - Ax) * (Y - Ay) - (By - Ay) * (X - Ax))
    
    
#cv2.imshow('cc', frame_res)
#cv2.imshow('bb', frame_canny)
#cv2.waitKey()
#cv2.destroyAllWindows()