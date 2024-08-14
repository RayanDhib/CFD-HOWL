# CFD-HOWL: Computational Fluid Dynamics High-Order Writer Library

## Overview
CFD-HOWL (Computational Fluid Dynamics High-Order Writer Library) is an advanced software library designed to efficiently write high-order solution data generated by high-order computational fluid dynamics (CFD) methods. These methods include Flux Reconstruction (FR), Discontinuous Galerkin (DG), Spectral Difference (SD), among others. 

The library addresses the limitations of traditional data storage techniques, which often rely on artificial mesh subdivision and linear interpolation, leading to potential compromises in solution integrity and increased data storage. CFD-HOWL innovatively utilizes mesh geometric upgrades and writes using high-order elements, leveraging the capabilities of the CGNS (CFD General Notation System) format to maintain the fidelity of the computational results while ensuring compatibility with modern visualization tools such as ParaView and Tecplot that support high-order elements up to Q4.

## Key Features
- **High-Order Element Support**: Directly writes high-order elements, avoiding the inaccuracies associated with mesh subdivision.
- **Efficiency in Data Storage**: Significantly reduces data size and improves I/O efficiency by maintaining the high-order data integrity throughout the storage process.
- **Enhanced Post-Processing Time**: Optimizes the time required for post-processing by maintaining data in high-order formats suitable for direct use in advanced visualization software.

## Dependencies
- **[pyCGNS](https://github.com/pyCGNS/pyCGNS)**: Utilized for handling CGNS file operations within Python.
- **[NumPy](https://numpy.org/)**: Fundamental package for scientific computing with Python.
- **[SciPy](https://scipy.org/)**: Utilized for handling CGNS file operations within Python.

## Installation and Usage

CFD-HOWL can be installed and used in two ways:

### 1. [Using Docker](https://github.com/RayanDhib/CFD-HOWL/wiki/Docker-Installation-Guide)
Docker provides an isolated environment to run CFD-HOWL without worrying about dependencies or system configuration.

### 2. [Manual Installation](https://github.com/RayanDhib/CFD-HOWL/wiki/Manual-Installation-Guide)
For users who prefer not to use Docker, manual installation instructions are also available.

For detailed installation instructions, usage examples, and troubleshooting tips, please refer to our [Wiki](https://github.com/RayanDhib/CFD-HOWL/wiki).

## Contributing
We welcome contributions to CFD-HOWL. Please feel free to fork the repository, make improvements, and submit pull requests. For significant changes, please first open an issue to discuss what you would like to change.

## License
CFD-HOWL is open-source under the GNU Lesser General Public License v3.0 (LGPL-3.0). This license allows you to modify and redistribute the software under certain conditions. For more details, see the LICENSE file.

## Acknowledgments
- This project was initially developed as part of enhancements to the **COOLFluiD** platform, a world-class object-oriented HPC platform for CFD. COOLFluiD provides a comprehensive environment for implementing and running numerical simulations, with a focus on scalability, performance, and usability. For more information, visit [COOLFluiD on GitHub](https://github.com/andrealani/COOLFluiD).

For more detailed information or support, please open an issue in this repository.