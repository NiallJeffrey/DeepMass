# DeepMass
## Cosmological map inference with deep learning
[![arXiv](https://img.shields.io/badge/arXiv-1908.00543-b31b1b.svg)](https://arxiv.org/abs/1908.00543) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

DeepMass was developed for infering dark matter maps from weak gravitational lensing measurements, and uses deep learning to reconstruct cosmological maps.


![DeepMass_result](https://github.com/NiallJeffrey/DeepMass/blob/main/DES_mass_maps_demo/plots/DeepMass_result.jpg)


DeepMass can be incorporated into a Moment Network (see [Solving high-dimensional parameter inference: marginal posterior densities & Moment Networks](https://arxiv.org/abs/2011.05991)) for high-dimensional likelihood-free inference:


![CMB_readme_fig](https://github.com/NiallJeffrey/DeepMass/blob/main/CMB_foreground_demo/CMB_readme_fig.jpg)


([Single frequency CMB B-mode inference with realistic foregrounds from a single training image](https://arxiv.org/abs/2111.01138) Jeffrey et al. 2021 MNRAS Letters)

### Installation

To download data associated with the demos, this repository uses Git Large File Storage (git-lfs): https://git-lfs.github.com/

If this is not installed locally, the downloaded repository will include code but not data.

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
