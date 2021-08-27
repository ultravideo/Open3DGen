import copy 
import numpy as np
import pyrealsense2 as rs
import cv2
import open3d as o3d

pipeline = rs.pipeline()
config = rs.config()
# config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 60)
# config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 60)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

profile = pipeline.start(config)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

align_to = rs.stream.color
align = rs.align(align_to)

# decimation_filter = rs.decimation_filter()
temporal_filter = rs.temporal_filter()
holefilling_filter = rs.hole_filling_filter()
spatial_filter = rs.spatial_filter()

colors = []
depths = []
depths_filtered = []

for i in range(60*2):
    frames = pipeline.wait_for_frames()

every_i = 2
frame_i = 0
l_i = 0

blur_thr = 120


while True:
    frames = pipeline.wait_for_frames()

    aligned_frames = align.process(frames)
    aligned_depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    if not aligned_depth_frame or not color_frame: continue

    depth_image_unfiltered = np.asanyarray(aligned_depth_frame.get_data())

    # aligned_depth_frame = decimation_filter.process(aligned_depth_frame)
    aligned_depth_frame = spatial_filter.process(aligned_depth_frame)
    aligned_depth_frame = temporal_filter.process(aligned_depth_frame)
    
    depth_image = np.asanyarray(aligned_depth_frame.get_data())
    # depth_image = cv2.resize(depth_image, (848, 480))
    color_image = np.asanyarray(color_frame.get_data())

    mblur_variance = cv2.Laplacian(color_image, cv2.CV_64F).var()

    if mblur_variance >= blur_thr:
        depths_filtered.append(copy.deepcopy(depth_image))
        colors.append(copy.deepcopy(color_image))
        depths.append(copy.deepcopy(depth_image_unfiltered))
        l_i += 1
        print("added", str(l_i))

    frame_i += 1

    # Render images
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
    # depth_image_unfiltered_cmap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image_unfiltered, alpha=0.03), cv2.COLORMAP_JET)
    images = np.hstack((color_image, depth_colormap))
    cv2.namedWindow("img", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("img", images)
    key = cv2.waitKey(1)
    
    if key & 0xFF == ord('q') or key == 27:
        cv2.destroyAllWindows()
        break

pipeline.stop()

# exit(0)

for i in range(len(colors)):
    d = o3d.geometry.Image(depths[i].astype(np.uint16))
    df = o3d.geometry.Image(depths_filtered[i].astype(np.uint16))
    c = o3d.geometry.Image(colors[i].astype(np.uint8))

    # o3d.io.write_image("intr_1280/rgb_" + str(i) + ".png", c)

    o3d.io.write_image("capture/rgb/rgb_" + str(i) + ".png", c)
    o3d.io.write_image("capture/depth/depth_" + str(i) + ".png", d)
    # o3d.io.write_image("depth_filtered/depthfiltered_" + str(i) + ".png", df)

    print("written", i, "out of", len(colors))
