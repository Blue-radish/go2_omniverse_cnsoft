from setuptools import setup

package_name = 'paddle_ros2'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ziyueg@unitree',
    maintainer_email='ziyue.gao@foxmail.com',
    description='Paddle ROS2 bridge node',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ocr_node             = paddle_ros2.ocr_node:main',
            'pseudo_llm_converter = paddle_ros2.pseudo_llm_converter:main',
            'ernie_llm_converter  = paddle_ros2.ernie_llm_converter:main',
            'go2_motion_demo      = paddle_ros2.go2_motion_demo:main',
        ],
    },
)
