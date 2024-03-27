import math

import cv2
import pyrealsense2 as rs
import numpy as np


class DepthCamera:
    def __init__(self,clipping_distance_in_meters=0.5):
        self.depth_frame = None
        self.colorizer = rs.colorizer()
        self.pipeline = rs.pipeline()
        self.align = rs.align(rs.stream.color)
        config = rs.config()
        self.color_intrin = None
        #the model i use need this size
        image_w,image_h = 640,480
        config.enable_stream(rs.stream.depth, image_w, image_h, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, image_w, image_h, rs.format.bgr8, 30)
        # Start streaming
        profile = self.pipeline.start(config)
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        print("Depth Scale is: ", depth_scale)
        if 0 < clipping_distance_in_meters < 10:
            self.clipping_distance = clipping_distance_in_meters / depth_scale
        else:
            self.clipping_distance = np.inf


    def get_frame(self):
        frames = self.pipeline.wait_for_frames()
        frames = self.align.process(frames)
        color_frame = frames.get_color_frame()
        aligned_depth_frame = frames.get_depth_frame()
        self.depth_frame = aligned_depth_frame
        self.color_intrin = color_frame.profile.as_video_stream_profile().intrinsics

        depth_color_frame = self.colorizer.colorize(aligned_depth_frame)

        depth_data = np.asanyarray(aligned_depth_frame.get_data())

        color_image = np.asanyarray(color_frame.get_data())
        depth_color_image = np.asanyarray(depth_color_frame.get_data())
        if self.clipping_distance != np.inf:
            black = 0
            depth_data_3d = np.dstack((depth_data,depth_data,depth_data))
            color_image = np.where((depth_data_3d > self.clipping_distance) | (depth_data_3d <= 0), black, color_image)
            depth_data = np.where((depth_data > self.clipping_distance) | (depth_data <= 0), black, depth_data)
            depth_color_image = np.where((depth_data_3d > self.clipping_distance) | (depth_data_3d <= 0), black, depth_color_image)
        if not aligned_depth_frame or not color_frame:
            return False, None, None

        distance = self.calculate_distance(5)
        print("distance")

        return True, depth_data, color_image,depth_color_image

    def release(self):
        self.pipeline.stop()

    def calculate_distance(self, m):
        color_intrin = self.color_intrin
        width, height = color_intrin.width, color_intrin.height
        # Calculate the size of each division
        division_width = width // m
        division_height = height // m

        distances = []

        # Iterate through each division
        for i in range(m):
            for j in range(m):
                # Calculate starting and ending points for the current division
                start_x, end_x = i * division_width, (i + 1) * division_width
                start_y, end_y = j * division_height, (j + 1) * division_height

                # Calculate distances between points within the current division
                division_distances = []
                for x1 in range(start_x, end_x):
                    for y1 in range(start_y, end_y):
                        udist = self.depth_frame.get_distance(x1, y1)
                        for x2 in range(start_x, end_x):
                            for y2 in range(start_y, end_y):
                                vdist = self.depth_frame.get_distance(x2, y2)
                                point1 = rs.rs2_deproject_pixel_to_point(color_intrin, [x1, y1], udist)
                                point2 = rs.rs2_deproject_pixel_to_point(color_intrin, [x2, y2], vdist)
                                dist = math.sqrt(
                                    math.pow(point1[0] - point2[0], 2) + math.pow(point1[1] - point2[1], 2) + math.pow(
                                        point1[2] - point2[2], 2))
                                division_distances.append(dist)

                # Aggregate distances calculated within the current division
                distances.append(sum(division_distances) / len(division_distances))

        return distances
