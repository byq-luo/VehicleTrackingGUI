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
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
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
    The unmatched detections and trackers returned ought to be deleted from the original matrices as they are no longer necessary
    :param detections: the detections returned from the yolo net
    :param trackers: the list of trackers maintained by the previous stage of kalman filter
    :param iou_threshold: the threshold which is used to tweak false predictions
    :return: matches, unmatched detections, unmatched trackers
    """

    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)
    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
    """
    Calculating the cost matrix for linear assingment
    """
    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            iou_matrix[d, t] = iou(det, trk)
    """
    Linear Assignment chooses the most efficient set of groups given a set of possible cost values, in the cost matrix. Please google for further clarity
    """
    matched_indices = linear_assignment(-iou_matrix)

    """
    Calculating unmatched detections that have not been mapped to trackers
    """
    unmatched_detections = []
    for d, det in enumerate(detections):
        if (d not in matched_indices[:, 0]):
            unmatched_detections.append(d)

    """
    Calculating unmatched trackers that have not been mapped to detections
    """

    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if (t not in matched_indices[:, 1]):
            unmatched_trackers.append(t)

    # filter out matched with low IOU
    """
    Filtering detections and trackers that are below the IOU threshold. 
    """
    matches = []
    for m in matched_indices:
        if (iou_matrix[m[0], m[1]] < iou_threshold):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if len(matches) == 0:
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
        self.vehicle_count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
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
        self.SECONDS_TO_CALCULATE_AVERAGE = 2 * 60
        self.MOVING_AVG_WINDOW = 6 * 60
        self.prev_car_c_neg = 0
        self.prev_car_c_pos = 0
        self.prev_bus_c_neg = 0
        self.prev_bus_c_pos = 0
        self.prev_motorbike_c_neg = 0
        self.prev_motorbike_c_pos = 0
        """
        Always specify velocity in positive values. The code takes care of inter changing the sign
        """
        self.POSITIVE_VELOCITY_THRESHOLD = 1
        self.NEGATIVE_VELOCITY_THRESHOLD = 5

    def update(self, dets, box_results):
        """
        Requires: this method must be called once for each frame even with empty detections.
        Returns the a similar array, where the last column is the object ID.
        NOTE: The number of objects returned may differ from the number of detections provided.*

        :param dets: a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        :param box_results: The bounding boxes for the detections
        :return:
        """

        if self.new_video:
            self.trackers == []
            self.frame_count = 0
            self.ch = []
            self.ch1 = []
            self.name = []
            self.vehicle_count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            self.car_c_neg = []
            self.bus_c_neg = []
            self.motorbike_c_neg = []
            self.car_c_pos = []
            self.bus_c_pos = []
            self.motorbike_c_pos = []
        self.frame_count += 1
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        """
        Y = mX + c 
        Calculated Slope(m) and Constant(c) of the line segment.  
        """
        slope_line = (self.line_coordinate[1] - self.line_coordinate[3]) / (
                self.line_coordinate[0] - self.line_coordinate[2])

        constant = self.line_coordinate[1] - (slope_line * self.line_coordinate[0])

        """
        Predicting the next state of the trackers using Kalman Filter
        """
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]

            if np.any(np.isnan(pos)):
                to_del.append(t)
        """
        Masked invalid method removes rows that have full 0 values. Refer to numpy docs for further clarification
        """
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))

        for t in reversed(to_del):
            self.trackers.pop(t)

        """
        Calling utility function to match detections with trackers. Refer to the method documentation for further clarification
        """
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks)

        """
        Update matched trackers with assigned detections, for already vehicles that are being tracked
        """

        for t, trk in enumerate(self.trackers):
            if t not in unmatched_trks:
                d = matched[np.where(matched[:, 1] == t)[0], 0]
                trk.update(dets[d, :][0])

        """
        Create and initialise new trackers for unmatched detections. These are the new objects detected. Adding them to allow tracking
        """

        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :])
            self.trackers.append(trk)
        i = len(self.trackers)

        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            dl = np.array(d[0:4])
            """
            If new trackers are found, assign them a new id
            """
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))  # +1 as MOT benchmark requires positive
            i -= 1
            """
            Remove Dead trackers
            """
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
        for t in range(len(ret)):
            a = ret[t]
            b = a[0]  # b contains the bounding box for the Detection along with class id and confidence
            c = b[5]  # Detection Id
            vehicle_vel = b[4]  # velocity of the detection
            de = b[1]  # upper left y coordinate
            cd = b[0]  # upper left x coordinate
            x1, y1, x2, y2 = b[0], b[1], b[2], b[3]
            """
            Calculating for Negative Velocity
            """
            if vehicle_vel < -self.NEGATIVE_VELOCITY_THRESHOLD:
                for full_info in box_results:
                    """
                    Checking if the Kalman prediction is within the permissible limits
                    """
                    if ((full_info[2]) - 30) <= de <= ((full_info[2]) + 30) and ((full_info[0]) - 90) <= cd <= ((full_info[0]) + 90):
                        global vehicle_name
                        vehicle_name = full_info[4]
                        break
                    else:
                        vehicle_name = 'nothing'
                if c not in self.ch1:
                    """
                    Checks if the line segment intersects the borders of the bounding box
                    """
                    if self.verify_line_intersection(slope_line, constant, x1, x2, y1, y2):
                        if vehicle_name == 'car':
                            self.vehicle_count[0] += 1
                            self.ch1.append(c)
                        if vehicle_name == 'bus':
                            self.vehicle_count[1] += 1
                            self.ch1.append(c)
                        if vehicle_name == 'motorbike':
                            self.vehicle_count[2] += 1
                            self.ch1.append(c)
            """
            Calculating for positive velocity.
            """
            if vehicle_vel > self.POSITIVE_VELOCITY_THRESHOLD:
                for k in range(len(box_results)):
                    full_info = box_results[k]
                    if (((full_info[2]) - 30) <= de <= ((full_info[2]) + 30) and ((full_info[0]) - 90) <= cd <= (
                            (full_info[0]) + 90)):
                        vehicle_name = full_info[4]
                        break
                    else:
                        vehicle_name = 'nothing'
                if c not in self.ch:
                    if self.verify_line_intersection(slope_line, constant, x1, x2, y1, y2):
                        if vehicle_name == 'car':
                            self.vehicle_count[4] = self.vehicle_count[4] + 1
                            self.ch.append(c)
                        if vehicle_name == 'bus':
                            self.vehicle_count[5] = self.vehicle_count[5] + 1
                            self.ch.append(c)
                        if vehicle_name == 'motorbike':
                            self.vehicle_count[6] = self.vehicle_count[6] + 1
                            self.ch.append(c)
            # self.vehicle_count[3] = avg_vel
            b = np.append(b, self.vehicle_count)
            a = np.array([b])
            ret[t] = np.array(a)
        """
        Appending count
        c_neg = Count Negative | in flow
        c_pos = Count Positive | out flow
        """
        avg_frame_count = self.SECONDS_TO_CALCULATE_AVERAGE * self.frame_rate
        if self.frame_count % avg_frame_count == 0 and self.frame_count >= avg_frame_count:
            """
            Comparing values from history to find the number of vehicles passed in 1 sec
            """
            self.car_c_neg.append((self.vehicle_count[0] - self.prev_car_c_neg))
            self.bus_c_neg.append(self.vehicle_count[1] - self.prev_bus_c_neg)
            self.motorbike_c_neg.append(self.vehicle_count[2] - self.prev_motorbike_c_neg)
            self.car_c_pos.append(self.vehicle_count[4] - self.prev_car_c_pos)
            self.bus_c_pos.append(self.vehicle_count[5] - self.prev_bus_c_pos)
            self.motorbike_c_pos.append(self.vehicle_count[6] - self.prev_motorbike_c_pos)

            """
            Updating history
            """
            self.prev_car_c_neg = self.vehicle_count[0]
            self.prev_bus_c_neg = self.vehicle_count[1]
            self.prev_motorbike_c_neg = self.vehicle_count[2]
            self.prev_car_c_pos = self.vehicle_count[4]
            self.prev_bus_c_pos = self.vehicle_count[5]
            self.prev_motorbike_c_pos = self.vehicle_count[6]

            if self.frame_count % (self.MOVING_AVG_WINDOW * self.frame_rate) == 0:
                """
                Moving Average Calculation
                """
                self.vehicle_count[7] = round(sum(self.car_c_neg) / self.car_c_neg.__len__()) if self.car_c_neg.__len__() > 0 else 0
                self.vehicle_count[8] = round(sum(self.bus_c_neg) / self.bus_c_neg.__len__()) if self.car_c_neg.__len__() > 0 else 0
                self.vehicle_count[9] = round(sum(self.motorbike_c_neg) / self.motorbike_c_neg.__len__()) if self.car_c_neg.__len__() > 0 else 0
                self.vehicle_count[10] = round(sum(self.car_c_pos) / self.car_c_pos.__len__()) if self.car_c_neg.__len__() > 0 else 0
                self.vehicle_count[11] = round(sum(self.bus_c_pos) / self.bus_c_pos.__len__()) if self.car_c_neg.__len__() > 0 else 0
                self.vehicle_count[12] = round(sum(self.motorbike_c_pos) / self.motorbike_c_pos.__len__()) if self.car_c_neg.__len__() > 0 else 0
                self.car_c_neg.pop(0)
                self.bus_c_neg.pop(0)
                self.motorbike_c_neg.pop(0)
                self.car_c_pos.pop(0)
                self.bus_c_pos.pop(0)
                self.motorbike_c_pos.pop(0)

        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))

    def verify_line_boundaries(self, x, y):
        """
        Utility function to check if the given point lies within the line segment co-ordinates
        :param x: x co-ordinate of the point
        :param y: y co-ordinate of the point
        :return: True if the given point lie within the line segment region
        """
        y_max = self.line_coordinate[1] if self.line_coordinate[1] > self.line_coordinate[3] else self.line_coordinate[3]
        y_min = self.line_coordinate[1] if self.line_coordinate[1] < self.line_coordinate[3] else self.line_coordinate[3]
        x_max = self.line_coordinate[2] if self.line_coordinate[2] > self.line_coordinate[0] else self.line_coordinate[0]
        x_min = self.line_coordinate[2] if self.line_coordinate[2] < self.line_coordinate[0] else self.line_coordinate[0]
        return (x_min <= x <= x_max) and (y_min <= y <= y_max)

    def verify_line_intersection(self, slope_line, constant, x1, x2, y1, y2):
        """
        Utility function to check if the bounding box intersects the line or not.
        :param slope_line: slope of line segment
        :param constant: constant in the equation of line segement
        :param x1: top left x co-ordinate of the bounding box
        :param x2: bottom right x co-ordinate of the bounding box
        :param y1: top left y co-ordinate of the bounding box
        :param y2: bottom right y co-ordinate of the bounding box
        :return:
        """
        minimum = 100
        for i in range(0, int(y2 - y1)):
            y = y1 + i
            y_temp = slope_line * x1 + constant
            diff = abs(y - y_temp)
            if diff < minimum and self.verify_line_boundaries(x1, y):
                minimum = diff
            y_temp = slope_line * x2 + constant
            diff = abs(y - y_temp)
            if diff < minimum and self.verify_line_boundaries(x2, y):
                minimum = diff
        for j in range(0, int(x2 - x1)):
            x = x1 + j
            y_temp = slope_line * x + constant
            diff = abs(y1 - y_temp)
            if diff < minimum and self.verify_line_boundaries(x, y1):
                minimum = diff
            y_temp = slope_line * x + constant
            diff = abs(y2 - y_temp)
            if diff < minimum and self.verify_line_boundaries(x, y2):
                minimum = diff
        return round(minimum) == 0
