from setuptools import setup

package_name = 'perception'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='student',
    maintainer_email='student@example.com',
    description='Colour detection for CMP9767 LIMO',
    license='TODO',
    tests_require=['pytest'],
    entry_points={
    'console_scripts': [
        'detector_basic = perception.detector_basic:main',
        'controller_vision = perception.controller_vision:main',
    ],
},

)
