"""
    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016 Alex Bewley alex@dynamicdetection.com

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
from __future__ import print_function

import numpy as np
from filterpy.kalman import KalmanFilter
from numba import jit
from sklearn.utils.linear_assignment_ import linear_assignment




@jit
def iou(bb_test, bb_gt):
    """
    Computes IUO between two bboxes in the form [x1,y1,x2,y2]
    """
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
              + (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh)
    return (o)


def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
      [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
      the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h  # scale is just area
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
      [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    # print(x)
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if (score == None):
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2., x[5]]).reshape((1, 5))
    else:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2., score]).reshape((1, 5))


class KalmanBoxTracker(object):
    """
    This class represents the internel state of individual tracked objects observed as bbox.
    """
    count = 0

    def __init__(self, bbox):
        """
        Initialises a tracker using initial bounding box.
        """
        # define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array(
            [[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array(
            [[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]])

        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if ((self.kf.x[6] + self.kf.x[2]) <= 0):
            self.kf.x[6] *= 0.0
        self.kf.predict()
        # print(self.kf.x)
        self.age += 1
        if (self.time_since_update > 0):
            self.hit_streak = 0
        self.time_since_update += 1
        # print(self.kf.x)
        self.history.append(convert_x_to_bbox(self.kf.x))
        # vel = self.history[0]
        # ab = vel[0]
        # ab = np.append(ab, self.kf.x[6])
        # vel = np.array([ab])
        ##self.history = np.array([vel])

        # self.history[-1] = np.array(vel)
        # print(self.history[-1])
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.1):
    """
    Assigns detections to tracked object (both represented as bounding boxes)

    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if (len(trackers) == 0):
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)
    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)

    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            iou_matrix[d, t] = iou(det, trk)
    matched_indices = linear_assignment(-iou_matrix)

    unmatched_detections = []
    for d, det in enumerate(detections):
        if (d not in matched_indices[:, 0]):
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if (t not in matched_indices[:, 1]):
            unmatched_trackers.append(t)

    # filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if (iou_matrix[m[0], m[1]] < iou_threshold):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if (len(matches) == 0):
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


#
# ch = []
# name = []
# rama = [0, 0, 0, 0, 0, 0, 0]
# car_c = []
# bus_c = []
# motorbike_c = []


class Sort(object):
    def __init__(self, max_age=1, min_hits=3):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.trackers = []
        self.frame_count = 0
        self.ch = []
        self.ch1 = []
        self.name = []
        self.rama = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.car_c_neg = []
        self.bus_c_neg = []
        self.motorbike_c_neg = []
        self.car_c_pos = []
        self.bus_c_pos = []
        self.motorbike_c_pos = []
        self.frame_width = 0
        self.frame_height = 0
        self.line_coordinate = []
        self.new_video = False
        self.frame_rate = 0

    def update(self, dets, garg):
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections.
        Returns the a similar array, where the last column is the object ID.

        NOTE: The number of objects returned may differ from the number of detections provided.*
        """

        if self.new_video:
            self.trackers == []
            self.frame_count = 0
            self.ch = []
            self.ch1 = []
            self.name = []
            self.rama = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            self.car_c_neg = []
            self.bus_c_neg = []
            self.motorbike_c_neg = []
            self.car_c_pos = []
            self.bus_c_pos = []
            self.motorbike_c_pos = []
        self.frame_count += 1
        #
        scale_x = self.frame_width / 1780
        scale_y = self.frame_height / 886

        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        # pos_vel = []
        #
        # global vel
        # global i
        # vel = 0
        # i = 0
        # print(trks,'hello')

        slope_line = (self.line_coordinate[1] - self.line_coordinate[3]) / (
                self.line_coordinate[0] - self.line_coordinate[2])

        constant = self.line_coordinate[1] - (slope_line * self.line_coordinate[0])

        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]

            # if(pos[4] <= 0):
            #
            #
            #   vel = vel + pos[4]
            #   i = i + 1

            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]

            if (np.any(np.isnan(pos))):
                to_del.append(t)
        # if(vel == 0):
        #     avg_vel = 0
        # else :
        #     avg_vel = vel / i
        #     #print(vel)
        # print(i)

        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))

        for t in reversed(to_del):
            self.trackers.pop(t)

        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks)
        # print(matched, unmatched_dets, unmatched_trks)

        # update matched trackers with assigned detections
        for t, trk in enumerate(self.trackers):
            if (t not in unmatched_trks):
                d = matched[np.where(matched[:, 1] == t)[0], 0]
                trk.update(dets[d, :][0])

        # create and initialise new trackers for unmatched detections

        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :])

            # print(trk)

            self.trackers.append(trk)

        i = len(self.trackers)
        # print(i)
        # print('ok1')
        # print(len(garg))
        # print('ok2')
        # print(len(dets))
        # print('ok3')

        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            dl = np.array(d[0:4])
            # print(dl)
            # (trk.time_since_update < 1) and
            # print(len(d))

            if ((trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits)):
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))  # +1 as MOT benchmark requires positive
            i -= 1
            # remove dead tracklet
            if (trk.time_since_update > self.max_age):
                self.trackers.pop(i)
        # d = []
        # print(len(ret)

        for t in range(len(ret)):
            a = ret[t]
            # print(len(ret))

            b = a[0]
            # print(b)
            # b contains the bounding box for the Detection along with class id and confidence
            c = b[5]  # Detection Id
            vehicle_vel = b[4]  # velocity of the detection
            de = b[1]  # upper left y coordinate
            cd = b[0]  # upper left x coordinate
            new_var = b[2]  # bottom right corner x
            line_eq_1_left = (de - slope_line * cd - constant)
            line_eq_1_right = (c - slope_line * new_var - constant)
            line_eq_2 = line_eq_1_left + (135 * scale_y)
            line_eq_3 = line_eq_1_left - (175 * scale_y)
            xyz = 35 * b[2] - b[3] - 35 * 500 + 300

            # # xyz = (de - (19/24)*cd)
            # if(vehicle_vel< -.5):
            #     for k in range(len(garg)):
            #         full_info = garg[k]
            #         if(((full_info[2]) - 30) <= de <= ((full_info[2]) + 30) and ((full_info[0]) - 30) <= cd <= ((full_info[0]) + 30)):
            #             global vehicle_name
            #             vehicle_name = full_info[4]
            #             break
            #         else: vehicle_name = 'nothing'
            #
            #     if(vehicle_name == 'car'):
            #         if((-(25*scale_y)<= line_eq_1 <=(25*scale_y)) and (all([c!=x for x in self.ch]))): #and (xyz >= 0) ):
            #
            #             print("It is counted")
            #             self.rama[0] = self.rama[0] + 1
            #
            #             self.ch.append(c)
            #     if(vehicle_name == 'bus'):
            #         if((-(10*scale_y) <= line_eq_2 <= (10*scale_y)) and (all([c!=x for x in self.ch]))): #and (xyz >= 0)):
            #             self.rama[1] = self.rama[1] + 1
            #             self.ch.append(c)
            #
            #     if(vehicle_name == 'motorbike'):
            #         if((-(100*scale_y)<= line_eq_3 <= (100*scale_y)) and (all([c!=x for x in self.ch]))): #and (xyz >= 0)):
            #             self.rama[2] = self.rama[2] + 1
            #             self.ch.append(c)
            # #else :
            #    rama[3] = 0

            if vehicle_vel < -5:
                # print(vehicle_vel)
                # print(c)
                for k in range(len(garg)):
                    full_info = garg[k]
                    if (((full_info[2]) - 30) <= de <= ((full_info[2]) + 30) and ((full_info[0]) - 90) <= cd <= (
                            (full_info[0]) + 90)):
                        global vehicle_name
                        vehicle_name = full_info[4]
                        break
                    else:
                        vehicle_name = 'nothing'
                x1, y1, x2, y2 = b[0], b[1], b[2], b[3]
                min = 100
                if c not in self.ch1:
                    # Checks if the line segment intersects the borders of the bounding box
                    for i in range(0, int(y2 - y1)):
                        y = y1 + i
                        y_temp = slope_line * x1 + constant
                        diff = abs(y - y_temp)
                        if diff < min:
                            min = diff
                        y_temp = slope_line * x2 + constant
                        diff = abs(y - y_temp)
                        if diff < min:
                            min = diff
                    for j in range(0, int(x2 - x1)):
                        x = x1 + j
                        y_temp = slope_line * x + constant
                        diff = abs(y1 - y_temp)
                        if diff < min:
                            min = diff
                        y_temp = slope_line * x + constant
                        diff = abs(y2 - y_temp)
                        if diff < min:
                            min = diff

                    if int(min) == 0:
                        if vehicle_name == 'car':
                            self.rama[0] += 1
                            self.ch1.append(c)
                        if vehicle_name == 'bus':
                            self.rama[1] += 1
                            self.ch1.append(c)
                        if vehicle_name == 'motorbike':
                            self.rama[2] += 1
                            self.ch1.append(c)
                # if (vehicle_name == 'car'):
                #     if (((500 <= new_var <= 1200) or (500 <= cd <= 800)) and (all([c != x for x in self.ch1]))):
                #         global rama
                #         self.rama[0] = self.rama[0] + 1
                #         # print(c)
                #         # print('car')
                #
                #         self.ch1.append(c)
                # if (vehicle_name == 'bus'):
                #     if ((500 <= new_var <= 700) and (all([c != x for x in self.ch1]))):
                #         self.rama[1] = self.rama[1] + 1
                #         # print(c)
                #         # print('bus')
                #
                #         self.ch1.append(c)
                #
                # if (vehicle_name == 'motorbike'):
                #     if ((500 <= new_var <= 1200) and (all([c != x for x in self.ch1]))):
                #         self.rama[2] = self.rama[2] + 1
                #         # print(c)
                #         # print('bike')
                #         # print(rama)
                #         # print(c)
                #
                #         self.ch1.append(c)
            if vehicle_vel > 1:
                # print(vehicle_vel)
                # print(c)
                for k in range(len(garg)):
                    full_info = garg[k]
                    if (((full_info[2]) - 30) <= de <= ((full_info[2]) + 30) and ((full_info[0]) - 90) <= cd <= (
                            (full_info[0]) + 90)):

                        vehicle_name = full_info[4]
                        break
                    else:
                        vehicle_name = 'nothing'
                x1, y1, x2, y2 = b[0], b[1], b[2], b[3]
                min = 100
                if c not in self.ch:
                    for i in range(0, int(y2 - y1)):
                        y = y1 + i
                        y_temp = slope_line * x1 + constant
                        diff = abs(y - y_temp)
                        if diff < min:
                            min = diff
                        y_temp = slope_line * x2 + constant
                        diff = abs(y - y_temp)
                        if diff < min:
                            min = diff
                    for j in range(0, int(x2 - x1)):
                        x = x1 + j
                        y_temp = slope_line * x + constant
                        diff = abs(y1 - y_temp)
                        if diff < min:
                            min = diff
                        y_temp = slope_line * x + constant
                        diff = abs(y2 - y_temp)
                        if diff < min:
                            min = diff
                    if int(min) == 0:
                        if (vehicle_name == 'car'):
                            self.rama[4] = self.rama[4] + 1
                            self.ch.append(c)
                        if (vehicle_name == 'bus'):
                            self.rama[5] = self.rama[5] + 1
                            self.ch.append(c)
                        if (vehicle_name == 'motorbike'):
                            self.rama[6] = self.rama[6] + 1
                            self.ch.append(c)
                # if (vehicle_name == 'car'):
                #     # if(((500 <= cd <= 800) or (500 <= new_var <= 800)) and (all([c!=x for x in self.ch]))):
                #     if (((400 <= b[3] <= 500) and (b[2] > 500) and (all([c != x for x in self.ch])))):
                #         self.rama[4] = self.rama[4] + 1
                #         # print(c)
                #         # print('car')
                #
                #         self.ch.append(c)
                # if (vehicle_name == 'bus'):
                #     if ((500 <= cd <= 700) and (all([c != x for x in self.ch]))):
                #         # self.rama[5] = self.rama[5] + 1
                #         # print(c)
                #         # print('bus')
                #
                #         self.ch.append(c)
                #
                # if (vehicle_name == 'motorbike'):
                #     if ((1000 <= cd <= 1400) and (all([c != x for x in self.ch]))):
                #         # self.rama[6] = self.rama[6] + 1
                #         # print(c)
                #         # print('bike')
                #         # print(rama)
                #         # print(c)
                #
                #         self.ch.append(c)

            # self.rama [3] = avg_vel
            b = np.append(b, self.rama)
            a = np.array([b])
            ret[t] = np.array(a)
        self.car_c_neg.append(self.rama[0])
        self.bus_c_neg.append(self.rama[1])
        self.motorbike_c_neg.append(self.rama[2])
        self.car_c_pos.append(self.rama[4])
        self.bus_c_pos.append(self.rama[5])
        self.motorbike_c_pos.append(self.rama[6])

        avg_frame = 20 * self.frame_rate
        real_var = self.frame_count - avg_frame

        if (real_var >= 0):
            self.rama[7] = self.car_c_neg[int(avg_frame) - 1] - self.car_c_neg[0]
            self.rama[8] = self.bus_c_neg[int(avg_frame) - 1] - self.bus_c_neg[0]
            self.rama[9] = self.motorbike_c_neg[int(avg_frame) - 1] - self.motorbike_c_neg[0]
            self.rama[10] = self.car_c_pos[int(avg_frame) - 1] - self.car_c_pos[0]
            self.rama[11] = self.bus_c_pos[int(avg_frame) - 1] - self.bus_c_pos[0]
            self.rama[12] = self.motorbike_c_pos[int(avg_frame) - 1] - self.motorbike_c_pos[0]
            self.car_c_neg.pop(0)
            self.bus_c_neg.pop(0)
            self.motorbike_c_neg.pop(0)
            self.car_c_pos.pop(0)
            self.bus_c_pos.pop(0)
            self.motorbike_c_pos.pop(0)

        if (len(ret) > 0):
            return np.concatenate(ret)
        return np.empty((0, 5))

# def parse_args():
#    """Parse input arguments."""
#    parser = argparse.ArgumentParser(description='SORT demo')
#    parser.add_argument('--display', dest='display', help='Display online tracker output (slow) [False]',action='store_true')
#    args = parser.parse_args()
#    return args

##if __name__ == '__main__':
#  # all train
#  sequences = ['PETS09-S2L1','TUD-Campus','TUD-Stadtmitte','ETH-Bahnhof','ETH-Sunnyday','ETH-Pedcross2','KITTI-13','KITTI-17','ADL-Rundle-6','ADL-Rundle-8','Venice-2']
#  args = parse_args()
#  display = args.display
#  phase = 'train'
#  total_time = 0.0
#  total_frames = 0
#  colours = np.random.rand(32,3) #used only for display
#  if(display):
#    if not os.path.exists('mot_benchmark'):
#      print('\n\tERROR: mot_benchmark link not found!\n\n    Create a symbolic link to the MOT benchmark\n    (https://motchallenge.net/data/2D_MOT_2015/#download). E.g.:\n\n    $ ln -s /path/to/MOT2015_challenge/2DMOT2015 mot_benchmark\n\n')
#      exit()
#    plt.ion()
#    fig = plt.figure()

#  if not os.path.exists('output'):
#    os.makedirs('output')

#  for seq in sequences:
#    mot_tracker = Sort() #create instance of the SORT tracker
#    seq_dets = np.loadtxt('data/%s/det.txt'%(seq),delimiter=',') #load detections
#    with open('output/%s.txt'%(seq),'w') as out_file:
#      print("Processing %s."%(seq))
#      for frame in range(int(seq_dets[:,0].max())):
#        frame += 1 #detection and frame numbers begin at 1
#        dets = seq_dets[seq_dets[:,0]==frame,2:7]
#        dets[:,2:4] += dets[:,0:2] #convert to [x1,y1,w,h] to [x1,y1,x2,y2]
#        total_frames += 1

#        if(display):
#          ax1 = fig.add_subplot(111, aspect='equal')
#          fn = 'mot_benchmark/%s/%s/img1/%06d.jpg'%(phase,seq,frame)
#          im =io.imread(fn)
#          ax1.imshow(im)
#          plt.title(seq+' Tracked Targets')

#        start_time = time.time()
#        trackers = mot_tracker.update(dets)
#        cycle_time = time.time() - start_time
#        total_time += cycle_time

#        for d in trackers:
#          print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1'%(frame,d[4],d[0],d[1],d[2]-d[0],d[3]-d[1]),file=out_file)
#          if(display):
#            d = d.astype(np.int32)
#            ax1.add_patch(patches.Rectangle((d[0],d[1]),d[2]-d[0],d[3]-d[1],fill=False,lw=3,ec=colours[d[4]%32,:]))
#            ax1.set_adjustable('box-forced')

#        if(display):
#          fig.canvas.flush_events()
#          plt.draw()
#          ax1.cla()

#  print("Total Tracking took: %.3f for %d frames or %.1f FPS"%(total_time,total_frames,total_frames/total_time))
#  if(display):
#    print("Note: to get real runtime results run without the option: --display")
