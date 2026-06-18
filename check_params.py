import pyrealsense2 as rs

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
profile = pipeline.start(config)

# get intrinsics
intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

fx = intr.fx
fy = intr.fy
cx = intr.ppx
cy = intr.ppy

print(f"fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")

camera_params = (fx, fy, cx, cy)

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
profile = pipeline.start(config)

# check what device is connected
device = profile.get_device()
print(f"Device: {device.get_info(rs.camera_info.name)}")
print(f"Serial: {device.get_info(rs.camera_info.serial_number)}")
print(f"Firmware: {device.get_info(rs.camera_info.firmware_version)}")

intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
print(f"fx={intr.fx:.2f}, fy={intr.fy:.2f}, cx={intr.ppx:.2f}, cy={intr.ppy:.2f}")
print(f"Resolution: {intr.width}x{intr.height}")
