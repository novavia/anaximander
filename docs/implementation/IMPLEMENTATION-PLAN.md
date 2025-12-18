# ANAXIMANDER IMPLEMENTATION PLAN

## Overview

This document is intended as a dynamic, iteratively updated implementation plan for the Anaximander project. Its role is to link the architectural and design documents with the project management toolkit, comprised of epics and issues. Given the magnitude of the project, the implementation plan provides a forward-looking reference that serves as a compass as well as a way to document major scope and schedule updates. In time, it is also intended to become a repository for tooling, methods and workflow practices that have project-wide applicability. It is also expected that a project governance document will become necessary if and when contributors join the project, at which point the articulation between governance and implementation guidelines will require further refinements.

Its current structure is as follows:

* The tooling section describes the development tools and methods used for the project
* High-level project milestones -these are externally visible accomplishments tied to a functional scope
* Sequence of implementation -the sequence of implementation is more inward-facing and a precursor to the development of epics and issues

## Tooling

Anaximander is developed in Python 3.14, which will be the minimal Python version supported by project.

The repository is hosted on GitHub.

The initial development will be conducted in VS Code, using GitHub Copilot for AI-assisted development.

### Docstring Conventions and Workflow

This project uses **Google-style Python docstrings** for all public APIs.
Docstrings are treated as part of the public contract and are used to
generate reference documentation.

---

#### 1. Scope

Docstrings are **required** for:
- Public modules
- Public classes
- Public functions and methods
- Decorators and metaprogramming constructs
- User-facing DSL / compiler / runtime interfaces

Docstrings are **optional** for:
- Private helpers (`_leading_underscore`)
- Trivial glue code
- Pure implementation details

---

#### 2. Style: Google Docstrings

##### General Rules
- Use triple double quotes (`"""`)
- First line: short, imperative summary
- Blank line after summary
- Describe **semantics and contract**, not implementation
- Type information belongs in type hints, not in the docstring

---

##### Function / Method Template

```python
def example(arg1: Type, arg2: Type | None = None) -> ReturnType:
    """
    One-line summary of what the function does.

    Optional longer explanation clarifying semantics, constraints,
    or important invariants.

    Args:
        arg1: Description of the argument.
        arg2: Description of the argument.

    Returns:
        Description of the return value.

    Raises:
        SomeError: Conditions under which this error is raised.
    """
```

##### Class Template

```python
class Example:
    """
    One-line summary of the class responsibility.

    Longer description explaining the role of the class in the system,
    its lifecycle, and any important invariants.
    """
```

##### Decorator Template

```python
def example_decorator(obj: T) -> T:
    """
    Modify a class or function to enable a specific behavior.

    Describes what is altered semantically, when the modification
    takes effect, and what guarantees (or lack thereof) are provided.

    Args:
        obj: The object being decorated.

    Returns:
        The modified object.
    """
```

---

#### 3. Content Guidelines

Docstrings SHOULD:
* Explain what the API guarantees
* Clarify semantic intent
* State important constraints or expectations
* Mention backend- or compiler-dependent behavior explicitly if relevant

Docstrings SHOULD NOT:

* Repeat the function name
* Describe line-by-line implementation
* Promise behavior not enforced by the code
* Encode internal algorithms

---

#### 4. Consistency

* Use the same terminology as the design documents
* Prefer project vocabulary over generic terms
* Avoid mixing docstring styles (NumPy / reST)

#### 5. Documentation Generation (Future)
This convention is compatible with:
* MkDocs
* mkdocstrings
* mkdocstrings-python
* Sphinx (via napoleon)

## Project Milestones

The initial project milestones are as follows:

* Establish the Anaximander Modeling Language (AML) as a Python library. With this milestone, it will be possible to develop an ontology in AML, using a subset of the language. The focus is on creating an internally coherent declarative library that implements the core constructs of AML.
* Compile AML to SQLAlchemy for PostgreSQL and SQLite. This milestone will mark the first usable version of the Anaximander project, offering the ability to instantiate a database from AML declarations.
* Establish the Digital Twin Interface. This milestone adds compilation to interface classes (`nxtype` and `Interface`), closing the loop from model specification to a system interface built around that same model. At this point, our hypothesis is that this milestone constitutes a minimal viable product (MVP) that can be employed to build data platforms.

Moving beyond this MVP, we foresee the next milestones as follows, with the exact ordering to be determined:

* Implementation of an externally-facing query application programming interface (API). This milestone involves model compilation to FastAPI or an equivalent library. The API will be broken down into a read-only data output port, and an administrative interface for CRUD operations on entities, documents and data artifacts.
* Implementation of parsing and validation -assumed to be initially left out from the MVP
* Implementation of aggregate archetypes and interfaces -for collections and dataframes. This milestone involves compilation to Pandera models.
* Lakehouse interface: with this milestone, it will become possible to store records in a structured data lake, using a combination of the Iceberg and/or DuckLake open formats.
* In-memory digital twin: this milestone will offer the ability to load a running Python process with a slice of data -essentially a range query in the entity-temporal-spatial domain, and keep it synchronized with durable storage.

Further still, but critically, the next set of milestones will tackle data processing. This involves specifications that have yet to be written, in particular the ability to express data transformations in AML. Along with this, the project will develop compilation capabilities to enable the service architecture, particularly data ingestion, the data mesh, and the event-driven architecture.

Additional cross-functional milestones will be weaved into this fabric, particularly:

* Model versioning will become important to support data platform implementation and will be introduced as soon as necessary
* Tracked code patches provide the ability to modify compiled code and reapply the modification upon every subsequent compilation
* A data permission model that accounts for both multi-tenancy and role-based access control

## Sequence of Implementation

As of this version, the sequence of implementation focuses on the first milestone, which is to reach an internally coherent AML library. The tentative sequence of implementation unfolds as follows:

* Create a base `Arche` class, from which all archetypes inherit
* Create the `Protodescriptor` base class
* Create the `Prototype` metaclass
* Create the `Archetype` metaclass and the `archetype` decorator
* Add traits and the the `trait` decorator
* Implement the base `Object` archetype
* Develop dedicated namespaces for functions (`dt`, `math`, `str` and `geo`), and for metadescriptors (`meta`, `option`, and `nxfield`)
* Populate the protodescriptor and archetype hierarchies