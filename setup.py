import setuptools

setuptools.setup(
    name             = 'wrapyfi',
    version          = '0.4.7',
    description      = 'Wrapyfi is a wrapper for simplifying Middleware communication',
    url              = 'https://github.com/fabawi/wrapyfi/',
    author           = 'Fares Abawi',
    author_email     = 'fares.abawi@outlook.com',
    maintainer       = 'Fares Abawi',
    maintainer_email = 'fares.abawi@outlook.com',
    packages         = setuptools.find_packages(),
    install_requires = ['pyyaml>=5.1.1', 
                        # 'opencv-python>=4.2.0.34', 
                        'opencv-contrib-python>=4.2.0.34'],
    python_requires  = '>=3.6',
    setup_requires   = ["cython>=0.29.1", "numpy>=1.19.2"]
)
