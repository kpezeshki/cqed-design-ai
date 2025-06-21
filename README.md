# cqed-design-ai
Collection of MCP servers for automated design and characterization of planar superconducting circuits. For the 2025 MCP x Quantum Science Hackathon.

# Internal planning

Copied from slack messages yesterday.

#### project goal: cad generation through qiskit metal and resonator frequency. kappa targeting through sonnet, all automated by LLM.

Let's update (this doc)[https://docs.google.com/document/d/1B-r4muDJw9WNXbEdJYd3dghIXhSP0lSPmKcZgdmk9Hk/edit?tab=t.0] in real time as we make progress. In summary, we need to build and prompt engineer for the following MCP servers:

- [x] MCP server (1) to (a) calculate coplanar capacitance -> coupling q given the width and spacing of the capacitive coupling structure and the feedline (b) calculate the resonant frequency of a quarter-wave structure given some overall length.
- [ ] MCP server (2) that acts as a ‘black box’ capacitive coupler simulation. This sets up and runs in sonnet an S31 simulation to extract Qc as described in section 4.2 of this https://web.physics.ucsb.edu/~bmazin/Papers/2008/Gao/Caltech%20Thesis%202008%20Gao.pdf. The simulation should include touchstone processing and return 1. A number 2. A plot of S31 against frequency.
- [ ] MCP server (3) that generates a single resonator coupled to a feedline given resonator length, resonator-feedline spacing, resonator-feedline coupler length. This server should also return the box size for simulation, the port locations, and the size of the smallest feature (so the model can intelligently choose the cell size).
- [ ] MCP server (4) that runs a sonnet simulation given a gds, box size, cell size, port locations, and saves the touchstone s2p file in some location and with some filename provided by the LLM.
- [ ] MCP server (5) that takes the s2p file and does a circle fit to extract Qc, f. This should also plot the data and the fit for the model to look at.
