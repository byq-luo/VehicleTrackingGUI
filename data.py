import numpy as np
import cv2
from datetime import datetime


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]
    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image
    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)
    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)
    # return the resized image
    return resized


FONT = cv2.FONT_HERSHEY_COMPLEX
FONT_SCALE = 0.4
FONT_COLOR = (0, 0, 0)
VIDEO_SCALE_RATIO = 0.5
RATIO_OF_BELOW_BOX = 0.35
date = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
OUTPUT_FILE_NAME = 'C:\\Users\\GRH1COB\\Desktop\\smartcity\\Smartcity\\tracking\\output_video\\{}.mp4'.format("test")
cap = cv2.VideoCapture("Video/test.mp4")
frame = 1
img_path = 'icons/bosch.png'
logo = cv2.imread(img_path, -1)
watermark = image_resize(logo, height=50)
watermark = cv2.cvtColor(watermark, cv2.COLOR_BGR2BGRA)
out = None

while frame is not None:
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
    frame = cv2.resize(frame, None, fx=VIDEO_SCALE_RATIO, fy=VIDEO_SCALE_RATIO, interpolation=cv2.INTER_LINEAR)
    width = frame.shape[1]
    height = frame.shape[0]
    b_height = round(frame.shape[0] * RATIO_OF_BELOW_BOX)
    blank_image = np.zeros((b_height, width, 3), np.uint8)
    blank_image[np.where((blank_image == [0, 0, 0]).all(axis=2))] = [240, 240, 240]
    overlay = np.zeros((height, width, 4), dtype='uint8')
    watermark_h, watermark_w, watermark_c = watermark.shape
    for i in range(0, watermark_h):
        for j in range(0, watermark_w):
            if watermark[i, j][3] != 0:
                offset = 10
                h_offset = height - watermark_h - offset
                w_offset = height - watermark_w - offset
                overlay[10 + i, 10 + j] = watermark[i, j]
    # text = cv::getTextSize(label, fontface, scale, thickness, & baseline);
    # cv::rectangle(im, or + cv::Point(0, baseline), or + cv::Point(text.width, -text.height), CV_RGB(0, 0,
    #                                                                                                 0), CV_FILLED);
    # text_size = cv2.getTextSize("DeepSense", FONT, FONT_SCALE, 1)
    # pt2 = ((width - int(width * 0.25)) + text_size.width, round(height * 0.1) + text_size.height)
    # cv2.rectangle(frame, (width - int(width * 0.25), round(height * 0.1)), pt2, (0, 0, 0),)
    cv2.putText(frame, "DeepSense", (width - int(width * 0.25), round(height * 0.1)), FONT, 1, (255, 255, 255), 2)
    cv2.addWeighted(overlay, 1, frame, 1, 0, frame)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    # Initial Data
    cv2.putText(blank_image, 'Vehicle Type', (30, 30), FONT, FONT_SCALE, FONT_COLOR, 1)
    cv2.putText(blank_image, 'Count (Y/N)', (30, 50), FONT, FONT_SCALE, FONT_COLOR, 1)
    cv2.putText(blank_image, 'Toll(Y/N)', (30, 70), FONT, FONT_SCALE, FONT_COLOR, 1)
    cv2.putText(blank_image, 'In Flow', (30, 90), FONT, FONT_SCALE, FONT_COLOR, 1)
    cv2.putText(blank_image, 'Out Flow', (30, 110), FONT, FONT_SCALE, FONT_COLOR, 1)
    cv2.putText(blank_image, 'Avg In Flow', (30, 130), FONT, FONT_SCALE, FONT_COLOR, 1)
    cv2.putText(blank_image, 'Avg Out Flow', (30, 150), FONT, FONT_SCALE, FONT_COLOR, 1)

    # Car Data:
    cv2.putText(blank_image, 'Car', (180, 30), FONT, FONT_SCALE, FONT_COLOR, 1)
    cv2.putText(blank_image, 'Yes', (180, 50), FONT, FONT_SCALE, FONT_COLOR, 1)
    cv2.putText(blank_image, 'No', (180, 70), FONT, FONT_SCALE, FONT_COLOR, 1)
    cv2.putText(blank_image, '2', (180, 90), FONT, FONT_SCALE, FONT_COLOR, 1)
    cv2.putText(blank_image, '0', (180, 110), FONT, FONT_SCALE, FONT_COLOR, 1)
    cv2.putText(blank_image, '3', (180, 130), FONT, FONT_SCALE, FONT_COLOR, 1)
    cv2.putText(blank_image, '1', (180, 150), FONT, FONT_SCALE, FONT_COLOR, 1)

    # Bus Data:
    cv2.putText(blank_image, 'Bus', (255, 30), FONT, FONT_SCALE, FONT_COLOR, 1)
    cv2.putText(blank_image, 'Yes', (255, 50), FONT, FONT_SCALE, FONT_COLOR, 1)
    cv2.putText(blank_image, 'No', (255, 70), FONT, FONT_SCALE, FONT_COLOR, 1)
    cv2.putText(blank_image, '2', (255, 90), FONT, FONT_SCALE, FONT_COLOR, 1)
    cv2.putText(blank_image, '0', (255, 110), FONT, FONT_SCALE, FONT_COLOR, 1)
    cv2.putText(blank_image, '3', (255, 130), FONT, FONT_SCALE, FONT_COLOR, 1)
    cv2.putText(blank_image, '1', (255, 150), FONT, FONT_SCALE, FONT_COLOR, 1)

    # Bike Data:
    cv2.putText(blank_image, 'Bike', (330, 30), FONT, FONT_SCALE, FONT_COLOR, 1)
    cv2.putText(blank_image, 'Yes', (330, 50), FONT, FONT_SCALE, FONT_COLOR, 1)
    cv2.putText(blank_image, 'No', (330, 70), FONT, FONT_SCALE, FONT_COLOR, 1)
    cv2.putText(blank_image, '2', (330, 90), FONT, FONT_SCALE, FONT_COLOR, 1)
    cv2.putText(blank_image, '4', (330, 110), FONT, FONT_SCALE, FONT_COLOR, 1)
    cv2.putText(blank_image, '2', (330, 130), FONT, FONT_SCALE, FONT_COLOR, 1)
    cv2.putText(blank_image, '1', (330, 150), FONT, FONT_SCALE, FONT_COLOR, 1)

    # Truck Data:
    cv2.putText(blank_image, 'Truck', (405, 30), FONT, FONT_SCALE, FONT_COLOR, 1)
    cv2.putText(blank_image, 'No', (405, 50), FONT, FONT_SCALE, FONT_COLOR, 1)
    cv2.putText(blank_image, 'No', (405, 70), FONT, FONT_SCALE, FONT_COLOR, 1)
    cv2.putText(blank_image, '2', (405, 90), FONT, FONT_SCALE, FONT_COLOR, 1)
    cv2.putText(blank_image, '0', (405, 110), FONT, FONT_SCALE, FONT_COLOR, 1)
    cv2.putText(blank_image, '0', (405, 130), FONT, FONT_SCALE, FONT_COLOR, 1)
    cv2.putText(blank_image, '0', (405, 150), FONT, FONT_SCALE, FONT_COLOR, 1)

    # Rickshaw Data:
    cv2.putText(blank_image, 'Three Wheeler', (480, 30), FONT, FONT_SCALE, FONT_COLOR, 1)
    cv2.putText(blank_image, 'Yes', (480, 50), FONT, FONT_SCALE, FONT_COLOR, 1)
    cv2.putText(blank_image, 'Yes', (480, 70), FONT, FONT_SCALE, FONT_COLOR, 1)
    cv2.putText(blank_image, '2', (480, 90), FONT, FONT_SCALE, FONT_COLOR, 1)
    cv2.putText(blank_image, '0', (480, 110), FONT, FONT_SCALE, FONT_COLOR, 1)
    cv2.putText(blank_image, '0', (480, 130), FONT, FONT_SCALE, FONT_COLOR, 1)
    cv2.putText(blank_image, '0', (480, 150), FONT, FONT_SCALE, FONT_COLOR, 1)

    # Tractor Data:
    cv2.putText(blank_image, 'Tractor', (600, 30), FONT, FONT_SCALE, FONT_COLOR, 1)
    cv2.putText(blank_image, 'Yes', (600, 50), FONT, FONT_SCALE, FONT_COLOR, 1)
    cv2.putText(blank_image, 'No', (600, 70), FONT, FONT_SCALE, FONT_COLOR, 1)
    cv2.putText(blank_image, '2', (600, 90), FONT, FONT_SCALE, FONT_COLOR, 1)
    cv2.putText(blank_image, '0', (600, 110), FONT, FONT_SCALE, FONT_COLOR, 1)
    cv2.putText(blank_image, '0', (600, 130), FONT, FONT_SCALE, FONT_COLOR, 1)
    cv2.putText(blank_image, '0', (600, 150), FONT, FONT_SCALE, FONT_COLOR, 1)


    img = np.row_stack((frame, blank_image))
    fheight = img.shape[0]
    fwidth = img.shape[1]
    if out is None:
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        fps = cap.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter(OUTPUT_FILE_NAME, fourcc, fps, (fwidth, fheight), True)
    else:
        out.write(img)
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
out.release()
cap.release()
