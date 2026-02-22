import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/mygo/go2_omniverse_cnsoft/go2_omniverse_ws/install/paddle_ros2'
