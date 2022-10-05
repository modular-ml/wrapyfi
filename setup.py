import setuptools

with open('requirements.txt') as f:
    requirements = f.read().splitlines()


setuptools.setup(
    name             = 'wrapify',
    version          = '0.4',
    description      = 'Wrapify is a wrapper for simplifying Middleware communication',
    url              = 'https://git.informatik.uni-hamburg.de/abawi/wrapify/',
    author           = 'Fares Abawi',
    author_email     = 'fares.abawi@uni-hamburg.de',
    maintainer       = 'Fares Abawi',
    maintainer_email = 'fares.abawi@uni-hamburg.de',
    packages         = setuptools.find_packages(),
    install_requires = requirements,
    python_requires='>=3.6',
)