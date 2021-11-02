# DeepMass
## Cosmological map inference with deep learning
[![arXiv](https://img.shields.io/badge/arXiv-1908.00543-b31b1b.svg)](https://arxiv.org/abs/1908.00543) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

DeepMass was developed for infering dark matter maps from weak gravitational lensing measurements, and uses deep learning to reconstruct cosmological maps.

### Installation
#### Using pip

```
!pip install 'git+https://github.com/NiallJeffrey/DeepMass.git'
```

#### From source
```
python setup.py install 
```
or for a cluster:

```
python setup.py install --user
```

### Prerequisites

```
Python 3; Tensorflow>=2.2; healpy
```

### Running the tests

```
python unit_tests.py
```

### Authors

* **Niall Jeffrey** 
* **Francois Lanusse** 

### License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
