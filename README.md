# Inhomogeneous Rashba Junction

## Set up the environment

Thhis module have been developed with `Python 3.8.2`, perhaps it should work fine as well for newer version.

Install requarements

```bash
pip install -r requirements.txt
```

All the examples and appliatation are run in Jupyther Notebooks. To oper the notebooks run in cutrrent directory

```bash
jupyter notebook
```

## Description

`RashbaJunction`-claass present hight level interfaces that allows to modify the properties of the system and compute the scattering matrix for given energy. Therefore first of all must it must be imported.

```python
from RachbaJunction import RachbaJunction
```

`RashbaJunction`-claass is instanciatd either with none parameters or with a place holder for the interface: a list of interfaces position and the corresponding SOC energy. In the case the magnetic fiels isn't homogeneous the third entry of the list should be a magnetic field profile (otherwice it set to be 1 implicitly).

* Initialize Junction-class with single interface and  E_so/E_z = 1 on both sides

```python
junction = RashbaJunction([[0], [1, 1]])
```

* Initialize Junction-class with double interface at k_z*x = \pm 1 and  E_so/E_z = 1 on the left and E_so/E_z = 2 on the rigth and pure Zeeman in center.

``` python
junction = RashbaJunction([[-1, 1], [1, 0, 2]])
```

* Initialize Junction-class with inhomogeneous magnetic field

``` python
junction = RashbaJunction([[0], [1, 1], [2, 4]])
```

**`Note:`**  that in the case the magnetic field is homogeneous the position of the parameters are expected to be dimensionless: the interfaces are expressed in term of Zeeman wave length x*k_Z on the other hand the SOC energy in term of Zeeman energy E/E_Z. Furthermore the sign of the SOC constant correspond to the sign of the SOC energy.

On the other hand in the case the magnetic fied is not homogeneous the parameters are assujmed to be dimensional(whoever are always set to be adimensional internaly). In this way the SOC energy must have the same units of the Zeeman energy instead the interfaces position should be exprtessed in term of sqrt(2 m /hbar^2)*x.

### Change the profile

The SOC energy profile can be changed with IndexError

```python
# set SOC energy on the right
junction[1] = -1
# set SOC energy on the left
junction[0] = 1
```

On the other hand the position of the interfaces must be acessed with the correspponding property

```python
# set the position of the left interface
junction.interface[0] = -1
# set the position of the rigth interface
junction.interface[1] = 1
```

### Compute the Scattering matrix

The scattering matrix for a given energy can be computed with

```python
S = junction.get_scattering_matrix(0.5)
```

This will return either a `ScatteringMatrix`-object or araise one of the error `EnergyOutOfRangeError`, `InsulatorError`. The first occure in the case in which for given energy there is not any physical solution i.e. E < -E_so * (1 + (E_Z/(2E_so))^2). the former instead occure if in one of the external region **only** evanescent modes are present and therefore the transmision can not be properly defined.
Below I will discuss how this erron can be handled easliy.

**`Note:`** as previously, here the energy is expressed in therm of Zeeman energy E/E_Z if the magnetic field is uniform and it is dimensional in the case the magnetic field is inhomogeneous.

### Scattering matrix

This will return a `ScatteringMatrix`-object that helpes to check if scatterring matrix is unitary

```python
S.is_unitary
```

and compute the transmission and reflection coefficients

```python
T = S.t_coef
R = S.r_coef
```

## Use in Jupyter notebook

It is  convinient to use the Jupyter notebook to run the computation and visualize the result.

from RashbaJunction.utilities import adjuct_Tick, make_grid, error_decorator
