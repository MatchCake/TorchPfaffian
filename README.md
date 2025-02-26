# TorchPfaffian: PyTorch-Based Pfaffian Computation

[![Star on GitHub](https://img.shields.io/github/stars/MatchCake/TorchPfaffian.svg?style=social)](https://github.com/MatchCake/TorchPfaffian/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/MatchCake/TorchPfaffian?style=social)](https://github.com/MatchCake/TorchPfaffian/network/members)
[![Python 3.6](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-310/)
[![downloads](https://img.shields.io/pypi/dm/MatchCake)](https://pypi.org/project/MatchCake)
[![PyPI version](https://img.shields.io/pypi/v/MatchCake)](https://pypi.org/project/MatchCake)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

![Tests Workflow](https://github.com/MatchCake/TorchPfaffian/actions/workflows/tests.yml/badge.svg)
![Dist Workflow](https://github.com/MatchCake/TorchPfaffian/actions/workflows/build_dist.yml/badge.svg)
![Doc Workflow](https://github.com/MatchCake/TorchPfaffian/actions/workflows/docs.yml/badge.svg)
![Publish Workflow](https://github.com/MatchCake/TorchPfaffian/actions/workflows/publish.yml/badge.svg)
![Code coverage](https://raw.githubusercontent.com/MatchCake/TorchPfaffian/coverage-badge/coverage.svg)


# Description

TorchPfaffian is a Python package for efficiently computing the Pfaffian of skew-symmetric matrices using PyTorch. 
Designed as a PyTorch-based alternative to [pfapack](https://github.com/basnijholt/pfapack), it enables GPU 
acceleration and supports automatic differentiation, making it particularly useful in physics, quantum computing, 
and machine learning applications.  

## Features:
- Efficient Pfaffian computation for skew-symmetric matrices  
- GPU acceleration via PyTorch  
- Support for automatic differentiation  
- Seamless integration with PyTorch tensors  




## Installation

With `python` and `pip` installed, run the following commands to install TorchPfaffian:
```bash
pip install torchpfaffian
```

With `poetry` installed, run the following commands to install TorchPfaffian:
```bash
poetry add torchpfaffian
```


# Important Links
- Documentation at [https://MatchCake.github.io/TorchPfaffian/](https://MatchCake.github.io/TorchPfaffian/).
- Github at [https://github.com/MatchCake/TorchPfaffian/](https://github.com/MatchCake/TorchPfaffian/).




# Found a bug or have a feature request?
- [Click here to create a new issue.](https://github.com/MatchCake/TorchPfaffian/issues/new)


## License
[Apache License 2.0](LICENSE)

## Acknowledgements


## Citation
Repository:
```
@misc{torchpfaffian_Gince2025,
  title={Torch Pfaffian},
  author={Jérémie Gince},
  year={2025},
  publisher={Université de Sherbrooke},
  url={https://github.com/MatchCake/TorchPfaffian},
}
```
