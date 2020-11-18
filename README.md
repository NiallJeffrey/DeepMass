# DeepMass
See the paper: "Deep learning dark matter map reconstructions from DES SV weak lensing data"

This code makes uses Keras with Tensorflow to create a CNN to reconstruct dark matter maps from noisy and incomplete weak lensing data.

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
Python 3 (version 3.6 works)

Tensorflow (version 1.12.0 works)

Keras (version 2.2.4 works)
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
