import setuptools


def check_cv2(default_python="opencv-python"):
    UPGRADE_CV2 = False
    REQUIRED_CV2_VERSION = "4.2.0"
    try:
        import pkg_resources
        from packaging import version
        import cv2

        if version.parse(cv2.__version__) < version.parse(REQUIRED_CV2_VERSION):
            UPGRADE_CV2 = True
            raise ImportError(f"OpenCV version must be at least {REQUIRED_CV2_VERSION}")
    except ImportError as e:
        import pkg_resources

        if UPGRADE_CV2:
            print(e, "Will try to upgrade OpenCV")
            if "opencv-python" in [p.project_name for p in pkg_resources.working_set]:
                additional_packages = [f"opencv-python>={REQUIRED_CV2_VERSION}"]
            elif "opencv-contrib-python" in [
                p.project_name for p in pkg_resources.working_set
            ]:
                additional_packages = [f"opencv-contrib-python>={REQUIRED_CV2_VERSION}"]
            elif "opencv-python-headless" in [
                p.project_name for p in pkg_resources.working_set
            ]:
                additional_packages = [
                    f"opencv-python-headless>={REQUIRED_CV2_VERSION}"
                ]
            else:
                raise ImportError(
                    f"Unknown OpenCV package installed. Please upgrade manually to version >={REQUIRED_CV2_VERSION}"
                )
        else:
            print(f"OpenCV not found. Will try to install {default_python}")
            additional_packages = [f"{default_python}>={REQUIRED_CV2_VERSION}"]
    else:
        print("OpenCV found. Will not install it")
        additional_packages = []
    return additional_packages


setuptools.setup(
    name="wrapyfi",
    version="0.5.0",
    description="Wrapyfi is a wrapper for simplifying Middleware communication",
    url="https://github.com/fabawi/wrapyfi/blob/main/",
    project_urls={
        "Documentation": "https://wrapyfi.readthedocs.io/en/latest/",
        "Source": "https://github.com/fabawi/wrapyfi/",
        "Tracker": "https://github.com/fabawi/wrapyfi/issues",
    },
    author="Fares Abawi",
    author_email="f.abawi@outlook.com",
    maintainer="Fares Abawi",
    maintainer_email="f.abawi@outlook.com",
    packages=setuptools.find_packages(),
    extras_require={
        "docs": ["sphinx", "sphinx_rtd_theme", "myst_parser"],
        "pyzmq": ["pyzmq>=19.0.0"],
        "numpy": ["numpy>=1.19.2"],
        "websocket": ["python_socketio>=5.0.4"],
        "zenoh": ["eclipse-zenoh>=1.0.0"],
        "mqtt": ["paho-mqtt>=2.0"],
        "headless": ["wrapyfi[pyzmq]", "wrapyfi[numpy]"]
        + check_cv2("opencv-python-headless"),
        "headless_websocket": ["wrapyfi[websocket]", "wrapyfi[numpy]"]
        + check_cv2("opencv-python-headless"),
        "headless_zenoh": ["wrapyfi[zenoh]", "wrapyfi[numpy]"]
        + check_cv2("opencv-python-headless"),
        "headless_mqtt": ["wrapyfi[mqtt]", "wrapyfi[numpy]"]
        + check_cv2("opencv-python-headless"),
        "complete": [
            "wrapyfi[numpy]",
            "sounddevice",
            "soundfile",
            "Pillow",
            "pandas",
            "wrapyfi[pyzmq]",
            "wrapyfi[websocket]",
            "wrapyfi[zenoh]",
            "wrapyfi[mqtt]",
            "wrapyfi[docs]",
        ]
        + check_cv2("opencv-contrib-python"),
        "all": ["wrapyfi[numpy]", "wrapyfi[pyzmq]"]
        + check_cv2("opencv-contrib-python"),
    },
    install_requires=["pyyaml>=5.1.1"],
    python_requires=">=3.6",
    setup_requires=["cython>=0.29.1"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Information Technology",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Human Machine Interfaces",
        "Topic :: System :: Distributed Computing",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows :: Windows 10",
        "Operating System :: Microsoft :: Windows :: Windows 11",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
