# FRAMEWORK ARCHITECTURE

##   Key Concepts

### Digital Twin

Anaximander adopts a broad and inclusive definition of digital twins: any software system that models and tracks the state of a physical-world environment qualifies as one. Under this view, even a simple data logger is a digital twin. At the other end of the spectrum, more complex systems may join data streams across heterogeneous assets, detect conditions and trigger alerts or actions, run simulations, or deploy machine learning models for predictive analytics.

Despite this range, most digital twins share three core characteristics:

- A static description of the physical environment through persistent entities (e.g., buildings, vehicles, machines) and their relationships.
- Acquisition of telemetry data from physical assets or external systems, often (but not necessarily) in real time.
- Representation of the system’s dynamic state — past, present, or hypothetical — as the primary function of the software.

In practice, the state of the environment may include raw sensor data, derived metrics, and synthetic indicators — from temperature readings to traffic congestion levels to computed safety scores.

This definition applies more naturally to *dynamic* digital twins, such as those used in IoT or robotics, where time-series telemetry is the primary currency. Other types of digital artifacts — like schematic models used in construction or design — may be called digital twins but lack the continuously evolving state that Anaximander is designed to support.

In dynamic settings, the digital twin’s core is the infrastructure that captures, computes, and stores evolving system states. Business applications — such as workflow automation or operational monitoring — are layered atop this core. Anaximander is purpose-built to define, deploy, and maintain this foundation, offering structured programming interfaces through which applications interact with the dynamic twin system.

### Functional Backbone

Anaximander supports a core set of digital twin system functions through a modular and declarative modeling approach:

- **Versioned entity storage** — Persist physical-world entities and their relationships, capturing attribute changes and relationship history over time. This enables time-travel queries and supports long-term system introspection.
- **Entity interface** — Expose CRUD operations and configuration management for entities and relationships through user-facing and programmatic interfaces.
- **Schema evolution** — Track and manage changes to the digital twin’s data model over time. Anaximander supports non-destructive structural updates, enabling systems to evolve without losing compatibility or historical continuity.
- **Data ingestion** — Capture telemetry and external system data through streaming and batch connectors.
- **State update pipelines** — Perform incremental updates to the digital twin’s state using high-throughput transformation and aggregation pipelines.
- **Event-driven triggers** — React to sparse, rule-defined events via concurrent reactive pipelines. Ideal for alerts, notifications, and automated workflows.
- **Time-aware storage** — Manage telemetry and derived state across lifecycle stages, including recent-data caches and long-term archives.
- **Data publication** — Expose curated state via REST and streaming APIs for external consumers and downstream systems.
- **Historical and virtual replays** — Enable simulations, calibrations, what-if analyses, and planning scenarios by replaying historical data, injecting synthetic inputs, and varying twin attributes.
- **Observability** — Support logging and monitoring for physical events, system operations, user actions, and infrastructure state.
- **Developer interface** — Provide a Python-native programming surface optimized for interactive analysis and application development.

### Domain-Specific Modeling

Anaximander is a domain-specific modeling (DSM) framework for digital twins of physical-world infrastructure. It enables developers to specify high-level semantic models that are compiled into robust, modular system code. This modeling-first approach yields better organization, lower defect rates, and more consistent architecture than manually authored implementations.

At its core is the Anaximander Modeling Language (AML), a Python library that provides declarative constructs embedded in class definitions. AML is used to express the ontology of the target environment, including entities, relationships, telemetry structures, and transformation logic. While not a standalone language, AML defines a modeling layer that abstracts away boilerplate code and system-level details.

From this ontology, Anaximander generates Python-based system code organized around open-source and cloud client libraries. This includes data schemas, APIs, processing logic, and validation routines. Deployment requires a complementary configuration that specifies the cloud resources (e.g., time series stores, object stores, compute functions) and credentials used to instantiate a given system.

In addition to compilation, Anaximander provides a Digital Twin Interface (DTI): a Python-native API for Create, Read, Update, and Delete (CRUD) operations. The DTI exposes entity relationships as navigable objects and automatically adds properties and methods suited to the domain—for instance, computing spatial bounds for geometry data or plotting time series with units.

Anaximander’s domain-specific modeling applies not just to system inputs but also to the interface. Its abstractions reflect the core patterns of digital twin systems, including:

- The distinction between models of persistent entities and those representing telemetry or derived states.
- The prominence of spatial and temporal constructs, including the critical distinction between event time (in the physical world) and process time (in the digital system).
- The use of physical units, dimensions, and metrology-related metadata such as uncertainty or measurement levels.
- The centrality of incremental computation over streaming or micro-batch data, requiring semantics that tolerate missing, delayed, or out-of-order events.
- The need to store, access, and visualize historical and hypothetical data with structured metadata for simulation, analysis, and planning.

Anaximander emphasizes the separation between model and system implementation. This isolation enables modularity, simplifies system evolution, and supports parallel deployments for testing, experimentation, and production use—all sharing a common semantic foundation.

### Code Generation and Deployment

Anaximander generates both system and interface code from model declarations, as described in the [Domain-Specific Modeling](#domain-specific-modeling) section. While the framework can be used in a one-shot mode—where a set of models are defined, a starter codebase is generated, and development continues independently—this approach forfeits key benefits. In particular, it leads to repetitive manual work as the system evolves and prevents use of the Digital Twin Interface (DTI) for interactive access and maintenance.

Continued use of the framework depends on managing changes carefully. The generation and deployment workflow provides two key features:

1. **Editable generated code with patch tracking**
    Users can edit generated code to customize behavior—for example, modifying API endpoints beyond default semantics. Anaximander tracks these changes as patches relative to the generated baseline. When models are updated and code is regenerated, those patches are reapplied automatically. In cases where a conflict arises, it can be resolved using standard code editing tools, much like a version control merge.
2. **Schema evolution with migration support**
    Changes to model schemas often require updates to the underlying system. Some updates—such as adding nullable fields—can be applied directly, while others require data migration. Anaximander leverages external tools when appropriate: for instance, if models are compiled to SQLAlchemy classes, schema migrations can be performed using Alembic, which detects metadata changes and produces the corresponding SQL migration scripts.

The diagram below summarizes the code generation and deployment loop. It illustrates how patches are applied and preserved, how migration steps may be inserted before deployment, and how deployment configuration complements models by specifying cloud resources and runtime parameters.



![Codegen.drawio](images\Codegen.drawio.png)

### Type System

Anaximander’s type system is one of its most original and powerful features. It serves a dual purpose: it underpins the Anaximander Modeling Language (AML) used to define digital twin ontologies, and it provides a type-safe, object-oriented programming interface for data scientists and application developers interacting with the twin.

#### Prototype–Archetype Composition

At its core, the type system draws a clear line between two kinds of abstractions:

- **Prototypes**, which are declarative model definitions—one may think of it as a schema. 
- **Archetypes**, which define structural and behavioral interfaces—one may think of it as a base class, but implementation relies on composition rather than inheritance.

This mirrors the framework’s larger philosophy: modeling is kept distinct from implementation. Prototypes describe what the data is. Archetypes describe how it behaves in context.

Consider a telematics application tracking vehicle fleets. Each vehicle periodically emits a telemetry record: vehicle ID, timestamp, location, and a payload of system variables. In AML, this record schema is defined using a data-class-like declaration called `VehicleRecord`, inheriting from a base `Model` class. These are prototypes—abstract, declarative, and not meant for instantiation.

In real applications, data doesn’t arrive as isolated records. It comes in sequences, batches, or keyed collections—each of which implies a different interface. These structures are typically realized as dataframes, where the index shape and partitioning semantics (e.g., single-entity vs. multi-entity slices) inform the available operations. Archetypes formalize this relationship between structure and behavior. For example:

- A series of time-ordered records for one vehicle exposes a `timespan` property and a trajectory `plot` method.
- A mapping from vehicle IDs to most recent records provides summary statistics for payload columns and a scatter `plot` method.

These structural semantics are expressed by combining the `VehicleRecord` prototype with an **archetype**, specifically `DataSequence` or `DataMapping` in the above example. For instance, `DataSequence[VehicleRecord]` becomes a concrete class with interface methods derived from both its prototype and archetype. Likewise, while `VehicleRecord` is declared as a prototype, the system exposes a corresponding runtime object class for use in code—one that combines schema definition with methods for access and analysis through an object archetype.

This composability enables Anaximander to offer a highly expressive and strongly typed object interface, where behavior is tied not just to data schema but to structure and context.

#### Typing Data Transformations

The prototype–archetype system also drives the framework’s data transformation model. Archetypes imply expected input/output behavior: for example, grouping and sessionizing records typically yields a `DataSequence`.

This allows developers to write high-level, type-safe transformation logic that can be validated and reused. For instance, vehicle trips can be generated by specifying `VehicleRecord.group_by_key(Vehicle).sessionize("5T")` in the generation method of a `Trip` model class. Here, gaps longer than 5 minutes signal a new trip. This logic is defined in terms of the prototype, which serves as a proxy for the unbounded `VehicleRecord` dataset, but realized as a streaming transformation whose result is structured with an appropriate archetype. The result is a pipeline interface that retains semantic awareness—both of structure and intent.

#### Traits and Type Keys

Two additional mechanisms enhance the type system’s expressiveness:

- **Traits** bridge prototype features with archetype behavior. If `VehicleRecord` has a spatial field, the resulting object `DataSequence[VehicleRecord]` inherits from `SpatialDataSequence`, gaining spatial methods.
- **Type Keys** enable controlled polymorphism. For example, a `DataTile` class may represent spatial tiles at multiple zoom levels. Declaring `z_level` as a type key allows users to write `DataTile[2]` to refer to a concrete prototype for zoom level 2.

#### Metadata

Metadata classes are used for various descriptors — these include referential descriptors such as time zones and spatial reference systems, physical descriptors such as units, as well as other constructs to describe model schemas and system-level properties such as permissions and versioning.

These metadata classes sit primarily on the implementation side of the framework in order to embody the concepts that they represent. By contrast, the AML modeling language relies on purpose-built descriptors and textual inputs -many of which are turned to formal metadata at runtime.

#### Summary

Anaximander’s type system introduces a modeling-first, composable approach to both data representation and transformation. It empowers users to:

- Define semantic models decoupled from system implementation or interface
- Build intuitive and expressive code with strong typing and powerful abstractions
- Write safe, concise, and readable data pipelines that reflect domain-specific semantics
- Interact with data using familiar Python idioms, enriched by semantically-aware object types

This fusion of declarative modeling and runtime interactivity makes Anaximander uniquely suited for digital twin development, especially in data-rich, physically grounded domains.

##   Framework Components

### Anaximander Modeling Language (AML)

The Anaximander Modeling Language (AML) is a Python package that enables declarative definitions of data models, types, and transformation pipelines. Its syntax resembles the built-in `dataclasses` module in Python: data models are declared as classes with typed fields. However, AML extends far beyond dataclasses by supporting specifications of relationships between models, data transformations, multiple kinds of metadata, and physical-world semantics.

At first glance, AML may seem similar to object-relational mappers like Django ORM or SQLAlchemy. But it plays a different role: it is not a runtime ORM, but a *modeling interface* that feeds a compilation engine. AML declarations are used to generate system code, including SQLAlchemy models for database integration, along with interfaces and pipelines across the digital twin stack.

AML’s foundational building blocks are **archetypes**, **prototypes**, **traits** and **protodescriptors**:

- **Archetypes** are abstract base classes that map the superstructure of objects to their behavior. Here superstructure refers to shape (scalar, collection, dataframe...) as well as index (a dataframe of sequential records with the same key maps to a different archetype than a dataframe of  records with a mix of keys, for instance).
- A **prototype** is a declarative model class that is associated with a base archetype. It describes the structure and semantics of data at the schema level, and is not meant to be instantiated directly. Prototypes are compiled into concrete system classes and runtime interfaces.
- **traits** are mixin classes that complement archetypes with reusable features — particularly spatiotemporal attributes.
- A **protodescriptor** is a declarative attribute used within a prototype. It defines metadata, fields, relationships, or derived views.

Together, these concepts form a type metamodel that lets users describe physical-world data with precision and consistency. All AML prototypes inherit from the `Prototype` metaclass, and are grouped into a set of top-level base classes, notably:

- **`Data`** defines custom data types assignable to model fields.
  - `Measurement`, a key subclass, represents a numeric value with a physical unit and associated metrology metadata.
  - `Media`, another subclass, models large binary objects like images or audio, which may be either embedded or stored separately from structured data tables.
- **`Model`** defines structured, key-value data objects. It is further specialized as:
  - `Entity`: captures persistent, identifiable and stateful objects — such as physical assets, machines, or people.
  - `Record`: describes informational records, especially telemetry events, but also static lookups or metrics.
  - `Document`: defines nested, structured configurations. Documents are always owned by another model, and support domain-specific variation without fragmenting storage — for example, allowing different machine types to share a common schema.

To populate prototypes, AML defines a rich set of protodescriptor types, categorized as follows:

- **Metadescriptors** describe type-level attributes. For instance, the physical unit for `Measurement` subclasses is implemented as a metadescriptor.
- **Schema descriptors** qualify model structure, such as declaring primary keys, or defining field groups for reuse or query shorthand.
- **Field descriptors** define concrete data fields and inter-model relationships, including views and states.
- **Method descriptors** enable the definition of parsing and validation methods.

AML defines the entire modeling vocabulary for the framework. It performs first-pass validation to ensure declarations are consistent and complete. These declarations are then parsed and fed to the Anaximander compiler, which generates system code tailored to the underlying infrastructure — such as database models, pipeline definitions, and API schemas.

By design, AML cleanly separates **model declaration** from **runtime behavior** so engineers can focus on semantics while the framework handles the implementation.

### Services Architecture

Anaximander generates system code from AML declarations to build or update a digital twin system. The resulting system consists of storage and compute services—typically cloud-based, but also deployable in local environments. The architecture distinguishes between two levels of abstraction:

1. **Application components**, which define the logical building blocks of the twin (e.g., entity storage, event handling, data ingestion). These are implementation-agnostic and derived from the structure of the AML model.
2. **Implementation services**, which are the actual infrastructure elements (e.g., PostgreSQL, cloud functions, message queues) that realize those components.

This separation provides flexibility: a single application service might be implemented using multiple cloud services, or several application services might share infrastructure. For example, telemetry, entity, and document storage could all resolve to the same database. Conversely, a system might define separate storage for different regions or entity types. Anaximander handles these mappings at compile time, so developers can focus on intent rather than low-level architecture—unless they want to.

#### Application Components

Anaximander defines a core set of application components, which serve as the functional backbone of a digital twin:

- **Data Loader**: Responsible for ingesting telemetry data, either by polling external sources or subscribing to pushed updates.
- **Update Processor**: Handles high-throughput, incremental data updates. It transforms raw telemetry through a *data mesh plane*, using user-defined operations expressed in AML.
- **Event Processor**: Responds to discrete, rule-defined triggers. Events flow through an *event plane* and can be used to raise alerts, launch workflows, or trigger integrations.
- **Storage Services**:
  - **Cache DB**, **Records DB**, and **Archive DB** for telemetry, partitioned by recency,
  - **Entity DB** for persistent, stateful objects,
  - **Document DB** for structured specifications and binary content (e.g. images, audio).
- **Data Reporter**: Exposes model data via REST APIs for inspection or integration.
- **Stream Publisher**: Broadcasts live updates via websockets or other push mechanisms.
- **Admin Service** *(not shown in diagram)*: Manages users, entities, configurations, and system policies via a REST API or graphical interface.

The following diagram illustrates this architecture, highlighting the dual flows of telemetry updates and event handling. The *data mesh plane* supports continuous, high-throughput processing, while the *event plane* captures discrete triggers and reactions. The update pipeline can emit events into the event bus to bridge the two layers.

![Dataflow Architecture](images\Dataflow Architecture.png "Dataflow Architecture — illustrating the data mesh and event planes")

#### Implementation Services

Application services are realized by cloud-native or local infrastructure, chosen based on project scale, latency needs, and cost. Common implementation services include:

- **Relational Databases** (e.g., PostgreSQL, Google Cloud SQL): Used for entity, record, and spec storage.
- **Data Warehouses** (e.g., BigQuery) or **Data Lakes** (e.g., S3 with Iceberg): Used for large-scale analytical storage or long-term telemetry.
- **In-Memory Caches** (e.g., Arrow, Redis, Memorystore): Used for low-latency access to recent telemetry.
- **Streaming Pipelines** (e.g., Google Dataflow, Flink, dbt): Execute update logic for telemetry processing.
- **Event Infrastructure**:
  - **Cloud Functions** or equivalent: Execute event handlers.
  - **Message Brokers** (e.g., Pub/Sub, Kafka): Manage event routing.
- **Serverless App Platforms** (e.g., Cloud Run, AWS Lambda): Host REST APIs and stream publishers.
- **Logging and Monitoring** tools, often built into cloud platforms.

Logging is handled across four distinct categories:

- **Event logs**: Record real-world events, sequenced by event time.
- **Process logs**: Track internal operations—data updates, job execution, event triggers.
- **Usage logs**: Record user queries and administrative activity.
- **System logs**: Capture infrastructure-level events such as deployments or service health.

Finally, Anaximander supports **one-off processing jobs**—for example, backfills, data migrations, or simulations. These are executed via a batch interface that typically spins up ad-hoc instances of the same services used in the real-time system.

### Compilers

Anaximander compilers translate AML model declarations into executable system code. While the architecture allows for multi-language output, current efforts are focused entirely on **Python** as the target language. This decision reflects both practical and strategic priorities:

1. **Python is broadly accessible**, even to engineers outside traditional software development roles.
2. **Python dominates the relevant ecosystem**, including web services, cloud SDKs, and data science tooling—making it a pragmatic choice for digital twin systems.

Rather than compiling AML into a standalone runtime, Anaximander compilers *transpile* AML declarations into Python modules that interface with a rich set of libraries. These libraries fall into two categories:

- **Open-source system libraries**, such as:
  - `SQLAlchemy` for object-relational mapping,
  - `Marshmallow` for data parsing and serialization.
- **Cloud SDKs**, such as:
  - `google-cloud-bigquery` for defining and interacting with BigQuery tables,
  - `google-cloud-pubsub` for message routing,
  - or any other service-specific client used in deployment.

Each compiler targets a single library and is responsible for transforming AML-defined prototypes into valid, idiomatic code for that library.

Additionally, Anaximander compiles runtime interfaces for each prototype declared in AML. These runtime interfaces are particularly designed for rich interactive sessions, but they can also be used in production code to provide access to the system code.

#### Compilation Workflow

Compilers operate by combining **runtime introspection** with **static source parsing**:

- At compile time, AML modules are **imported dynamically**. This gives compilers access to their declared **prototypes** and **protodescriptors**, including fields, types, default factories, and behaviors.
- However, introspection alone is not enough. For example, default values defined as expressions (e.g., `lambda: datetime.utcnow()`) are already evaluated at runtime. To access the **original expressions**, compilers also parse AML modules using Python’s built-in `ast` module, which extracts the abstract syntax tree (AST).

This two-step process—introspection and AST parsing—provides both a **semantic** and **syntactic** view of the prototype.

#### Compiler Architecture

Each compiler is composed of:

1. A **compiler class** for the target library. This class:
   - interprets AML prototypes,
   - analyzes their AST nodes,
   - and decides how to render the final code.
2. A set of **Jinja templates** that define the code skeletons for the target library:
   - These templates include class definitions, methods, and syntax scaffolding.
   - The compiler class injects rendered values into the template, often as stringified logic blocks.

While the compiler class can delegate structure and formatting to templates, the framework encourages putting **nontrivial logic in Python**, not Jinja. This keeps templates simple, while maintaining full programmatic control over the transformation process.

#### Selective Compilation

Not every prototype is compiled for every target. The decision depends on:

- **System configuration**: If a project doesn’t use BigQuery, there’s no need to generate BigQuery table definitions.
- **Prototype semantics**: Some model types (e.g., `Document` instances stored in a data lake) don’t need SQLAlchemy mappings, whereas others (e.g., `Entity` models) do.

These decisions are handled through **metaprogramming** at the prototype level. AML uses decorators and metaclass flags to indicate which prototypes apply to which compilation targets. This allows compilers to dynamically determine what to include and what to skip.

### Digital Twin Interface (DTI)

In Anaximander, models declared in AML are *abstract*. A class like `Machine`, declared to represent factory equipment, is not directly instantiable. Along with system code, the Anaximander compiler generates a runtime interface of the `Machine` model. Together, the runtime interfaces and system code form the **Digital Twin Interface (DTI)**. Hence the term *DTI* is used to broadly refer to the collection of compiled model classes for a specific digital twin project (interfaces and underlying system code) and to the framework components that define their structure and behavior. The DTI is the **application programming interface** for interacting with a digital twin system.

#### Two Key Use Cases

1. **Interactive use**: Data scientists use the DTI to explore and manipulate twin data. Compiled models expose rich, object-oriented interfaces with built-in support for:
   - Serialization and deserialization
   - Physical unit conversions
   - Plotting and spatial operations
   - Model composition and validation
2. **Embedded use**: The DTI can also be used in production systems outside the Anaximander framework—such as web application backends, custom analytics pipelines, or ETL workflows. In this context, the DTI serves as a type-safe, expressive data access layer that integrates smoothly with external codebases.

#### Structure

As explained in the [Type System](#type-system) section, AML defines **archetypes** and **prototypes**: archetypes encode behavior, and prototypes describe schema and features. This original type system enables a happy marriage between structural inheritance and composition: declarations use inheritance, but implementation relies on composition, which enables data schemas to remain orthogonal to object superstructure and behavior.

At compilation, the DTI resolves archetype and prototype into an implementation class, plus a facade interface that is designed to enhance developer experience. Implementation classes are instances of the `nxtype` metaclass, and compose a generic implementation with model-specific system code. The developer-facing interface inherits from the base `Interface` class. It enables model composition with deferred evaluation, and delegates implementation to a matching `nxtype`.

#### Dynamic Class Generation

While the DTI is mostly generated at compile time, it can also be **extended dynamically at runtime**. This is enabled by Python’s dynamic class creation capabilities. For instance, calling `nx.List[Machine]` at runtime will generate a class representing a list of machine instances. This new class is enriched with properties and methods based on the `Machine` model's protodescriptors.

This dynamic typing system allows developers to write expressive, natural code with strong semantic guarantees—even in exploratory workflows.

When dynamic constructs become part of a production system, they can be **promoted to compiled code** via AML’s deployment configuration. Doing so:

- Locks in the interface for better maintainability
- Enables IDE support such as autocompletion and linting
- Reduces runtime overhead from dynamic construction

### Deployment Dialect

The final major component of the Anaximander framework is the **deployment dialect** — the mechanism by which developers specify how and where a digital twin system should be instantiated.

While AML defines model semantics and the compilers generate system code, the deployment dialect provides the **runtime context**: what resources to use, how to connect to them, and which environments to target (e.g. development, staging, production). Its role is to bridge declarative intent with physical deployment.

At its simplest, the deployment dialect may simply define a set of environment variables or resource identifiers — for example, a PostgreSQL connection URL or a cloud storage bucket path. But it can also provide more advanced configuration, including:

- **Service mapping**: explicitly binding application services (such as entity storage or event processing) to specific cloud implementations.
- **Environment-specific overrides**: enabling the same codebase to be deployed across multiple environments with minimal duplication.
- **Optional logic**: injecting custom behavior for service provisioning or bootstrapping.

The exact form of the deployment dialect is still evolving. Likely options include:

- **Configuration-first approach**: using simple declarative files such as YAML, TOML, or `.env` to define resource mappings.
- **Python-based modules**: offering a richer, programmable interface where needed — for example, to dynamically compute resource names, handle conditional logic, or integrate secrets management.
- **AML integration**: treating deployment specifications as first-class AML extensions, leveraging the same parsing and validation mechanisms.

A key design goal is to balance **simplicity for common cases** with **expressiveness for advanced needs**. For many projects, the deployment dialect may consist of just a single configuration file per environment. But the framework remains extensible — allowing teams to grow into more sophisticated patterns as their systems evolve.

##  Repository Structure

The Anaximander repository is organized to host the core Python packages of the framework along with supporting assets. The structure is modular and extensible, following common Python conventions while reflecting the architectural components of the framework.

Below is an outline of the top-level directory layout, with brief annotations:

```yaml
docs/                    # Project documentation
resources/               # Placeholder for assets (e.g. diagrams, sample data, research)
src/                     # Source code root
├── anaximander/         # Main framework package
│   ├── aml/             # AML: modeling language implementation
│   ├── compilers/       # Compilers and transpilation logic
│   │   └── templates/   # Jinja templates for target-specific code generation
│   │       ├── dataclasses/   # Templates for 'dataclasses' output
│   │       ├── pydantic/      # Templates for 'pydantic' output
│   │       ├── sqlalchemy/    # Templates for 'sqlalchemy' output
│   │       └── ...            # Additional compiler targets
│   ├── config/          # Configuration and setup
│   │   └── project_template/  # Scaffolding for new Anaximander projects
│   ├── deployment/      # Deployment dialect interface
│   ├── dti/             # Digital Twin Interface (compiled runtime classes)
│   ├── utils/           # General-purpose utilities
│   └── projects.py      # Entry point for managing and launching projects
scripts/                 # Standalone scripts not part of importable libraries
tests/                   # Unit and integration tests
user/                    # Local directory for user-defined code (untracked)
```

This structure allows clean separation between framework logic, deployment metadata, user-defined models, and generated outputs. It also supports plug-in extensibility — for example, adding new compilers or deployment targets with minimal disruption to core modules.

##  Framework Operationalization

### Anaximander Projects

Users interact with the Anaximander framework through **projects** — self-contained code repositories that define a digital twin system. An Anaximander project imports the core `anaximander` package and follows certain organizational conventions to enable compilation, deployment, and execution.

A single Anaximander project may give rise to multiple deployed **instances** that share the same models and interface code, while differing in runtime configuration, deployment targets, or cloud environments.

#### Project Creation

Projects can be created manually — either by reorganizing an existing repository or starting from scratch. However, the recommended approach is to use the project template included in the Anaximander repository. This template scaffolds a valid project structure with minimal effort.

The framework provides project creation routines in the `projects.py` module. These routines will eventually be exposed via a command-line interface to streamline the developer workflow.

#### Project Structure

Below is the canonical structure of an Anaximander project:

```yaml
<project>/              # Top-level project directory (should match the project name)
├── .nxp                # Project metadata file (Anaximander Project file)
├── config/             # Configuration files and deployment resources
├── docs/               # Project documentation (partially auto-generated)
├── src/                # Project source code
│   ├── apps/           # Application logic (e.g. API endpoints, dashboards)
│   ├── <project>/      # Project-specific package (the digital twin interface)
│   │   └── api/        # Compiled system code
│   │       ├── sqlalchemy/     # Generated SQLAlchemy code
│   │       └── ...              # Additional compilation targets
│   ├── domain/         # AML model declarations (types, models, metadata)
│   └── scripts/        # Executable scripts not part of libraries
├── tests/              # Unit and integration tests
└── user/               # Untracked space for user experiments or local code
```

Key features include:

- **`.nxp` file**: This file declares the repository as an Anaximander project and stores metadata such as project name, version, and default configurations.

- **Package naming**: For consistency, the project name should match both the repository folder and the importable Python package that defines the Digital Twin Interface (DTI). For example, in a project named `my_factory`, importing `Machine` would look like:

  ```python
  from my_factory import Machine
  ```

- **Compiled system code**: Each compiler target (e.g. SQLAlchemy, Marshmallow) generates code under the `api` subpackage. These modules may be used directly if low-level access or performance optimizations are needed.

- **Prototypes vs. compiled artifacts**: The `domain` directory contains the canonical AML declarations. These are the source of truth for model definitions. The compiled code in `api` is generated from these declaration.

### Development Workflow

The Anaximander development workflow revolves around a clear separation between model declarations, compiled artifacts, and deployment configuration. Much of this process has been outlined in the [Code Generation and Deployment](#code-generation-and-deployment) section. Here, we contextualize that flow within the project structure introduced in the previous section, clarifying the role of each code directory during the development lifecycle.



![Development Workflow](images\Dev Workflow.drawio.png)

The typical workflow unfolds as follows:

1. **Model Development**
    The developer creates or modifies data models using AML in the `domain` directory. These are abstract declarations, intended to capture the structure, semantics, and transformations of domain-specific data.

2. **Compilation**
    When the project is compiled, the framework:

   - Parses the `domain` directory,
   - Validates declarations for internal consistency,
   - Generates both system code and a runtime digital twin interface.

   The results are placed into the `api` subdirectory within the project package:

    - Digital Twin Interface (e.g., `Machine`, `Sample`): High-level, user-friendly classes for interactive use.
   - Compiler targets (e.g., `sqlalchemy`, `marshmallow`): System code for persistence, serialization, etc.

3. **Interface Usage**
    Developers primarily interact with the digital twin via the compiled interface — e.g., importing domain objects like `Machine` or `Sample` and calling methods defined by their archetypes. This interface abstracts away system implementation details, enabling rapid, expressive development.

4. **Advanced Access**
    For specialized needs — e.g., low-level performance optimization or accessing non-standard features — developers may also use the generated system code directly.

By design, this workflow separates **model iteration** from **runtime code**, enabling safe, repeatable, and testable deployments. Models can evolve without disrupting existing instances until changes are deliberately compiled and pushed.