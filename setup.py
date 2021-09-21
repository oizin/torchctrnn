from distutils.core import setup
setup(
  name = 'torchctrnn',       
  packages = ['torchctrnn'],   
  version = '0.0.1',     
  license='MIT',        
  description = 'Lightweight package for using neural ODE based continuous time RNNs and related methods with pytorch and torchdiffeq',   
  author = 'Oisin Fitzgerald',                   
  author_email = 'o.fitzgerald@unsw.edu.au',     
  url = 'https://github.com/oizin/torchctrnn',   
  download_url = 'https://github.com/oizin/torchctrnn/archive/v_001.tar.gz',   
  keywords = ['DEEP LEARNING', 'TIME SERIES'],   
  install_requires=['torch>=1.9.0','torchdiffeq>=0.2.2'],
  python_requires='~=3.7',
  classifiers=[
      'Development Status :: 3 - Alpha',
      'License :: OSI Approved :: MIT License',
      'Programming Language :: Python :: 3',
      "Topic :: Scientific/Engineering :: Artificial Intelligence",
  ],
)