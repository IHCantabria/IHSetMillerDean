
# IHSetMillerDean
Python package to run and calibrate Miller and Dean (2004) equilibrium-based shoreline evolution model.

## :house: Local installation
* Using pip:
```bash

pip install git+https://github.com/defreitasL/IHSetMillerDean.git

```

---
## :zap: Main methods

* [millerDean](./IHSetMillerDean/millerDean.py):
```python
# model's it self
MillerDean(E, dt, a, b, cacr, cero, Yini, vlt)
```
* [cal_MillerDean](./IHSetMillerDean/calibration.py):
```python
# class that prepare the simulation framework
cal_MillerDean(path)
```



## :package: Package structures
````

IHSetMillerDean
|
├── LICENSE
├── README.md
├── build
├── dist
├── IHSetMillerDean
│   ├── calibration.py
│   └── millerDean.py
└── .gitignore

````

---

## :incoming_envelope: Contact us
:snake: For code-development issues contact :man_technologist: [Lucas de Freitas](https://github.com/defreitasL) @ :office: [IHCantabria](https://github.com/IHCantabria)

## :copyright: Credits
Developed by :man_technologist: [Lucas de Freitas](https://github.com/defreitasL) @ :office: [IHCantabria](https://github.com/IHCantabria).
