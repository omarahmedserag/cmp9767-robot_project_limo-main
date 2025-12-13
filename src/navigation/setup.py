from setuptools import setup, find_packages

package_name = 'navigation'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(),   # 🔥 IMPORTANT FIX
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ros',
    maintainer_email='omarseraj17@gmail.com',
    description='Waypoint navigation using Nav2',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'waypoint_nav = navigation.waypoint_nav:main' ,
            'frontier_explore = navigation.frontier_explore:main',
        ],
    },
)

