import setuptools

setuptools.setup(
    name             = 'wrapyfi',
    version          = '0.4.5',
    description      = 'Wrapyfi is a wrapper for simplifying Middleware communication',
    url              = 'https://github.com/fabawi/wrapyfi/',
    author           = 'Fares Abawi',
    author_email     = 'fares.abawi@outlook.com',
    maintainer       = 'Fares Abawi',
    maintainer_email = 'fares.abawi@outlook.com',
    packages         = setuptools.find_packages(),
    install_requires = ["pyyaml", "opencv-contrib-python"],
    python_requires  ='>=3.6',
    setup_requires   = ["cython>=0.28", "numpy>=1.14.0"]
)
