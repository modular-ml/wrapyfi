import setuptools

with open('requirements.txt') as f:
    requirements = f.read().splitlines()


setuptools.setup(
    name             = 'wrapyfi',
    version          = '0.4.1',
    description      = 'Wrapyfi is a wrapper for simplifying Middleware communication',
    url              = 'https://github.com/fabawi/wrapyfi/',
    author           = 'Fares Abawi',
    author_email     = 'fares.abawi@outlook.com',
    maintainer       = 'Fares Abawi',
    maintainer_email = 'fares.abawi@outlook.com',
    packages         = setuptools.find_packages(),
    install_requires = requirements,
    python_requires='>=3.6',
)