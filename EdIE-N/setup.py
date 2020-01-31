from setuptools import setup, find_packages

setup(
      name='edien',
      packages=find_packages(include=['edien', 'edien.*']),
      version='0.1.5',
      author='Philip John Gorinski, Andreas Grivas',
      author_email='agrivas@ed.ac.uk',
      description='EdieN',
      license='BSD',
      keywords=['clinical concept recognition'],
      scripts=['bin/edien_eval',
               'bin/edien_train'],
      classifiers=[],
      # We are depending on dict order insertion (new in 3.7)
      # + we are using dataclasses
      python_requires='>=3.7',
      install_requires=['torch==1.3.0', 'mlconf==0.0.7'],
      tests_require=['pytest']
)
