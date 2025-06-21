# cqed-design-ai
Collection of MCP servers for automated design and characterization of planar superconducting circuits. For the 2025 MCP x Quantum Science Hackathon.

# Internal planning

Copied from slack messages yesterday.

#### project goal: cad generation through qiskit metal and resonator frequency. kappa targeting through sonnet, all automated by LLM.

Let's update this doc in real time as we make progress

project components:
- [ ] qiskit-metal MCP server + good decorators

- [ ] qiskit-metal prompt engineering (when this part is done, the LLM should be able to take a plaintext 'draw a resonator coupled to a feedline' and generate reasonable looking CAD + save to gds, even if dimensions aren't quite right)

- [ ] pysonnet MCP server + good decorators

- [ ] pysonnet prompt engineering (here, the LLM should be able to take a GDS, import into sonnet, set up ports, box, etc reasonably, run a simulation, get S21 data)

- [ ] S21 processing MCP. This is basically a lorentzian fit, plus some sanity check utilities

- [ ] Integration. Here we need to prompt engineer the whole system so the LLM can run the geometry gen -> sim -> process s21 -> modify geometry -> loop till meets some benchmark. This might require more MCP storing intermediate results, etc so the LLM can be convinced to move the system in the right direction.
