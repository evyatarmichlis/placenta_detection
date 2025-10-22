import math

import pyrealsense2 as rs
import numpy as np


class DepthCamera:
    def __init__(self):
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
        # if 0 < clipping_distance_in_meters < 10:
        #     self.clipping_distance = clipping_distance_in_meters / depth_scale
        # else:
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


        return True, depth_data, color_image,depth_color_image

    def release(self):
        self.pipeline.stop()


