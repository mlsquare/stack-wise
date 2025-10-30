# Framework Glue: Specification, ETL, Scheduling, and Orchestration

This document captures the intended role and boundaries of the framework and how it glues together data, models, and optimizers without owning concrete implementations.

## Purpose

- The framework is specification and glue, not a model zoo.
- It provides:
  - Contracts/specs (BatchSpec, EmbeddingSpec, RackSpec, HeadSpec, OptimizerSpec, DataSpec)
  - Adapters (pre/post boundaries, dtype/shape/precision bridges)
  - Factories (read config/YAML or dicts; assemble artifacts by delegating to external providers)
  - ETL orchestration (treat data, model, optimizer as artifacts that evolve over time)
  - Training machine (orchestrator) that binds Data ETL + Model ETL + Optimizer ETL across a schedule/curriculum
- Concrete architectures, blocks, heads, and optimizers live outside and are plugged in.

## Configuration and Code – Both First-Class

- Config (YAML/dict) defines experiments reproducibly (CLI-friendly, inheritance-friendly).
- Programmatic API enables advanced users to compose and extend in code.
- Factories accept both: pure configs or already-constructed objects.

## Time as a First-Class Concept

- Support both normalized time t ∈ [0,1] and discrete steps k ∈ {0..T}.
- A curriculum (PhaseSchedule) maps time/steps to phases, tasks, and regimes:
  - Data evolution (datasets/transforms/masking)
  - Model evolution (freeze/unfreeze, growth)
  - Optimizer evolution (lr/schedules/groups)
- The training machine “steps” in time; at each step it assembles a full state triplet (Data_t, Model_t, Optimizer_t) and runs training.

## Interfaces (shape only)

- Embedding-like:
  - forward(input_ids) → hidden
  - get_spec() → EmbeddingSpec
- Rack-like:
  - forward(hidden) → hidden
  - to_spec() → RackSpec
- Head-like:
  - forward(hidden) → outputs
  - get_spec() → HeadSpec
- Data loader:
  - get_batch_spec() → BatchSpec
  - iterable over batches
- BatchSpec compatibility check between model expectation and data loader.

## Providers and Factories

- External providers implement concrete architectures/optimizers/data.
- Factories in the framework:
  - Read config/YAML (or receive objects) and call provider factories
  - Wire artifacts via adapters
  - Validate specs and expose a uniform runtime model to the training machine

## Separation of Concerns

- Framework contains: specs, adapters, factories (glue), ETL orchestration, schedule/curriculum, validation, training machine.
- External packages contain: transformer blocks, heads, racks, layers, concrete optimizer code.

## Sanity/Smoke Checks

- End-to-end single-loss training on a static dataset (single phase).
- Curriculum demo: expose different parts of the data or evolve the model across phases/time.

## Migration/Refactor Guidance

- Refactor existing code; do not rewrite. Preserve good concepts:
  - Progressive stacks/racks/blocks
  - Adapters and config-first factories
  - ETL idea across planes
- Make “time” explicit and first-class across data/model/optimizer.
- Keep backward compatibility where reasonable; provide shims and examples.

## Example Entry Points (conceptual)

- Config-first (CLI):
  - framework loads experiment YAML → calls model/data/optimizer factories → wires adapters → runs training machine by schedule
- Code-first:
  - user builds model/data/optimizer objects → passes to training machine → optionally provides a custom curriculum

---

This is the guiding document for keeping the framework as pure glue/spec/orchestration, while allowing many external models and optimizers to plug in cleanly via adapters and factories.
