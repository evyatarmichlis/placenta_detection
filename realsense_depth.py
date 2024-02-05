import cv2
import pyrealsense2 as rs
import numpy as np


class DepthCamera:
    def __init__(self):
        self.colorizer = rs.colorizer()
        self.pipeline = rs.pipeline()
        self.align = rs.align(rs.stream.color)

        config = rs.config()

        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))

        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        align = rs.align(rs.stream.depth)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # Start streaming
        self.pipeline.start(config)

    def get_frame(self):
        frames = self.pipeline.wait_for_frames()
        frames = self.align.process(frames)
        color_frame = frames.get_color_frame()
        aligned_depth_frame = frames.get_depth_frame()

        depth_color_frame = self.colorizer.colorize(aligned_depth_frame)

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        depth_color_image = np.asanyarray(depth_color_frame.get_data())

        if not aligned_depth_frame or not color_frame:
            return False, None, None
        return True, depth_image, color_image,depth_color_image

    def release(self):
        self.pipeline.stop()

