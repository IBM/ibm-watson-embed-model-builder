# *****************************************************************#
# (C) Copyright IBM Corporation 2022.                             #
#                                                                 #
# The source code for this program is not published or otherwise  #
# divested of its trade secrets, irrespective of what has been    #
# deposited with the U.S. Copyright Office.                       #
# *****************************************************************#
"""Setup to be able to build model_image_builder library.
"""

# Standard
import glob
import os

# Third Party
import setuptools

# get version of library
COMPONENT_VERSION = os.getenv("COMPONENT_VERSION")
if not COMPONENT_VERSION:
    raise RuntimeError("COMPONENT_VERSION must be set")

# base directory containing watson_runtime (location of this file)
base_dir = os.path.dirname(os.path.realpath(__file__))

# read requirements from file
with open(os.path.join(base_dir, "requirements.txt")) as handle:
    requirements = handle.read().splitlines()


package_name = "model_image_builder"
setuptools.setup(
    name=package_name,
    author="IBM",
    version=COMPONENT_VERSION,
    license="Copyright IBM 2022 -- All rights reserved.",
    description="Tools for building images with Watson Core models",
    install_requires=requirements,
    packages=setuptools.find_packages(include=(f"{package_name}*",)),
    package_data={
        package_name: glob.glob(f"{package_name}/resources/**", recursive=True)
    },
    include_package_data=True,
)
