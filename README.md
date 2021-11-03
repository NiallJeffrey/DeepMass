# DeepMass
[![arXiv](https://img.shields.io/badge/arXiv-1908.00543-b31b1b.svg)](https://arxiv.org/abs/1908.00543) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
## Cosmological map inference with deep learning

DeepMass was developed for inferring dark matter maps from weak gravitational lensing measurements, and uses deep learning to reconstruct cosmological maps.

DeepMass can also be incorporated into a Moment Network (see [ArXiv:2011.05991](https://arxiv.org/abs/2011.05991)) enabling high-dimensional likelihood-free inference.
##
![DeepMass_result](https://github.com/NiallJeffrey/DeepMass/blob/main/DES_mass_maps_demo/plots/DeepMass_result.jpg)
(Dark matter mass map demo [DES_mass_maps_demo/Training_example](https://github.com/NiallJeffrey/DeepMass/blob/main/DES_mass_maps_demo/Training_example.ipynb))
##
![CMB_readme_fig](https://github.com/NiallJeffrey/DeepMass/blob/main/CMB_foreground_demo/CMB_readme_fig.jpg)

CMB result from [``Single frequency CMB B-mode inference with realistic foregrounds from a single training image'' Jeffrey et al. 2021 MNRAS Letters](https://arxiv.org/abs/2111.01138) 

(CMB Foreground demo [CMB_foreground_demo/MomentNetwork_foregrounds](https://github.com/NiallJeffrey/DeepMass/blob/main/CMB_foreground_demo/MomentNetwork_foregrounds.ipynb))

## Installation

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

#### Prerequisites

```
Python 3; Tensorflow>=2.2; healpy
```

#### Running the tests

```
python unit_tests.py
```

## Authors

* **Niall Jeffrey** 
* **Francois Lanusse** 

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
