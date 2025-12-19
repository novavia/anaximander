

# ANAXIMANDER TYPE SYSTEM

## Overview

### Scope

The Anaximander type system lies at the core of the Anaximander framework. It translates the overall objectives of the framework into a set of abstractions and implements those as Python classes, metaclasses and code templates. As is the case with most application frameworks, a detailed understanding of this infrastructure is not required for typical usage and a superficial grasp of the key concepts will often suffice. This document describes the design of the type system and is primarily intended for framework contributors or advanced users, but it can also serve as a resource that ties together the main abstractions and their rationales.

Anaximander is an open-source Domain Specific Modeling (DSM) framework designed to build, operate and maintain data platforms. It particularly targets applications that model physical world environments, such as IoT or robotic backends, which we broadly refer to as digital twins. Anaximander is Python-centric: both its declarative language and the application interfaces that it surfaces are materialized as Python libraries, and the framework itself is programmed in Python with support from Jinja templates. 

The DSM paradigm, as implemented by Anaximander, is summed up in the following figure:

![DSM3.drawio](images\DSM3.drawio.png)

In a nutshell, it can be described as follows:

* The digital twin ontology is captured in a set of class declarations, using the Anaximander Modeling Language (AML) library. These classes are *prototypes*.
* The ontology is compiled into system code in target libraries: these include, among others, object-relational mappings (SQLAlchemy), REST API models (FastAPI), dataframe API models (Ibis), as well as client code for cloud infrastructure deployment, including databases, API services and event processing and data transformation engines.
* Type compilation also produces concrete system interfaces collectively designated as the Digital Twin Interface (DTI). The DTI types map one-to-one with the prototypes declared in the ontology, and inherit from the base `Interface` class. `Interface` objects exposes the fields declared in AML, plus a few reserved attributes prefixed by 'nx'. In particular, the bare `.nx` attribute gives access to methods. `Interface.nx` exposes an `nxobject` type (if called from the class) or instance (if called from an instance). Let's clarify these notions through a simple example:

```python
# AML
import anaximander as nx

class Vehicle(nx.Entity):
    vin: str = nx.field(key=True)
    make: str = nx.field()
    model: str = nx.field()
    year: int = nx.field()
```

```python
# DTI
>>> from myapp import Vehicle  # This is a compiled class that inherits from Interface, not the class declared in AML
>>> vehicle = Vehicle.nx.create('123', 'Toyota', 'Corolla', 2018)
>>> vehicle.make
Data[str](Vehicle.make)  # vehicle.make is an Interface instance
>>> vehicle.make()  # Interfaces are evaluated by calling them
'Toyota'
>>> vehicle.nx.model()  # .nx.model (unrelated to Vehicle's model attribute) is the default materialization method for models
Vehicle(vin='123', make='Toyota', model='Corolla', year=2018) # This is a Pydantic Model instance
>>> type(vehicle.nx).__name__  # vehicle.nx exposes an nxobject of type VehicleNx
'VehicleNx'
>>> Vehicle.nx.prototype  # This is the model class declared in AML
```

* Compiled `nxobject` types expose tailored interfaces that are aware of the prototype's fields, e.g. `VehicleNx` is the `nxobject` class compiled from the `Vehicle` prototype, and accessed from the `Vehicle` Interface subclass. For method implementation, they rely on generic base interfaces (in the above example, `EntityNx`). These base interfaces ultimately call the compiled system code, such as on ORM class (accessible as `VehicleNx.orm` in the above example).
* Custom application code, say a visualization dashboard, can work with either the digital twin interface types, or make direct use of the system code through Python imports.

As a result, the type system's scope covers the following functional areas:

* Model and custom data type declarations that form the digital twin's ontology
* An object-oriented programming interface (`Interface`/`nxobject`) that provides access to a broad set of data manipulation functions
* Semantic constructs and templates that enable the automatic compilation of declarative types into system code

The rest of this document describes how the type system's design meets these functional objectives.

### Design Goals

The type system supports the overall objectives of the Anaximander framework with the following design goals:

#### Composability

A primary objective of the framework is to present a unified interface for multi-modal data (relational entities, tabular records, vectors, documents). This means that objects that are part of the digital twin interface present attributes that point to different kinds of data, and are materialized with different libraries. While the declarative interface relies on inheritance to map the ontology to a type hierarchy, the implementation relies on type composition to bring together bags of functionality.

#### Separation of Concerns

As already evidenced by the overview diagram, the framework cleanly separates system interfaces from their implementation. This principle sits at the core of the framework's concept: the expression of the digital twin's ontology is kept distinct from the systems that run the digital twin, and likewise, the programming interface presented to the developer relies on three tiers: the schematic tier, pulled from the ontology, the capability tier, embodied by abstract interfaces, and the implementation tier, which is the compiled system code.

#### Pythonicity

If people throw the word "pythonic" around at-will, can we reasonably speak of pythonicity? At any rate, if there is going to be a single, or maybe more realistically, a main pane of glass for the Anaximander system, it is your favorite Python IDE. As such, we pursue the following design goals:

* Provide a rich object-oriented semantic model that serves both to declare the digital twin's ontology and to interact with the underlying system and digital infrastructure, either programmatically or in an interactive Python Read-Evaluate-Print-Loop (REPL)
* Rely on best-of-breed libraries in the Python data ecosystem. Anaximander objects are only references to data. The data itself is always kept in structured storage, whether in the cloud, on disk or in memory, and is materialized through third-party libraries (e.g. Pydantic models, dataframes...)
* Keep the modeling namespace uncluttered so that developers are not restricted in naming type fields and attributes
* Offer a great developer experience with clear semantics, straightforward syntax and rich integration with development tools for type inference, code auto-completion, and exposure of methods' signature and documentation
* Finally, make the type system extensible so that both individual projects and the open-source community can broaden the scope of the framework

### Structure

The Anaximander type system is effectively comprised of three hierarchies:

- Data models are declared in AML by writing data classes that are instances of the `prototype` metaclass. Prototypes are abstract classes with no other function than to encapsulate modeling declarations that can be compiled or referenced at runtime.

* At the core of the framework, we find the `nxobject` base class and the corresponding `nxtype` base metaclass. This is where the methods of the digital twin interface are written. This class hierarchy defines the abstract interfaces that mediate between the concrete interfaces and system code.
* The high-level interface to the digital twin is realized with concrete interfaces that inherit from the base `Interface` class.

In practice, `Interface` instances own attributes that match their modeling declarations. For instance a concrete `Vehicle` model has attributes `make` and `model`. Other than that, the `Interface` namespace is kept virgin except for the single attribute `.nx` so that developers are not constrained in naming model attributes or domain-specific methods. `Interface` accesses system methods through that the `.nx` attribute, which exposes `nxobject` (for `Interface` instances) and `nxtype` (for `Interface` subclasses) instances. 

An essential concept of the type system is the `archetype`. You can think of archetypes as a combination of data structure and functionalities in the absence of a schema. For instance, an identifiable physical object like `Vehicle` implements the `Entity` archetype. A time series of temperature samples implements the `Series` archetype. Archetypes are formally defined as part of the declarative `prototype` hierarchy, but the concept cuts across all three type hierarchies. `archetype` is complemented by `trait`. Whereas archetype define structure, traits operate as mix-in classes that specify further constraints and behavior. For instance, the `sample` trait indicates that a record type is produced at regular or pseudo-regular time intervals, whereas the `event` trait indicates that a record type is also a time series, but features a sparse index. Archetype and traits are complemented by so-called metacharacters, which are keyword arguments that can further modulate object behavior. Hence an `nxtype` combines an archetype, traits and metacharacters. When a `prototype` is declared in the modeling language, it encapsulates these properties so that it can be readily compiled. In particular, model prototypes feature a field schema that is handled as a metacharacter. Code compilers similarly rely on archetype, traits and metacharacters to generate system code on a per-type, per-library basis.

All these concepts are clarified and further expanded in the remainder of the document.

## AML prototype System

The `prototype` metaclass sits at the root of the framework. Modeling declarations take the form of simple data classes that are instances of `prototype`. In the class body of a `prototype`, developers use so-called protodescriptors to declare attributes. This terminology highlights the facts that a) these descriptor-like objects are designed for prototypes, and b) that they are not descriptors in the sense of the Python descriptor protocol, but rather precursors to concrete descriptors -which are implemented on interface objects. Here is a tiny example:

```python
# AML
import anaximander as nx

class Product(nx.Model):
	name: str = nx.data()
    sku: str = nx.data(key=True)
```

The `Product` model uses the `data` protodescriptor to declare two fields. Note however that prototypes are not limited to models structured as key-value maps. They include custom data types, collections, tables (i.e. dataframes) and more. 

In order to generate system code and a digital twin interface, prototypes must be compiled. However they are still referenced in user code post-compilation, because they act as the single source of truth for model declarations. In particular, `Interface` is implemented as a generic type that takes a `prototype` as its type parameter. At runtime, `nxobjects` may introspect model schemas that are held as attribute of `prototype` classes. 

The `prototype` design revolves around archetypes and traits, which in turn dictate what protodescriptors a particular type may implement. The following subsections describe the semantics and implementation mechanics of this design. For a complete list of available archetypes, traits and protodescriptors, one may refer to the AML design document.

### Archetypes

Archetypes are base prototypes that govern type interface and behavior. At the most basic level, there is a `Data` archetype for data types, and a `Model` archetype for model types. `Data` types expose a single `data` attribute. `String` is an example of a data type that inherits from the `Data` archetype. By contrast, `Model` types expose a mapping of fields. `Model` is further subclassed into three essential model archetypes:

* `Entity` models concrete, durable elements of the physical twin: machines, people, buildings, rooms, etc. Entities relate to other entities and can also implement states. They are identified by a unique id that is automatically appended to their model, though they can also implement key fields as an alternate method of unique identification.
* `Record` models purely informational objects - telemetry records and their derivatives, lookup tables... These are equivalent to facts in data warehousing terminology. Unlike entities, records don't receive a unique id, and they are identified by their indexing fields -most typically a key and a timestamp, though other options are available. 
* `Document` is a model that is stored independently -in a data lake or a document store for instance. Documents are typically used for specification or configuration, and more generally as dependent models that don't require identifying fields because they are always owned as attributes of other models. As a result, they are identified through a unique path, which by default is composed of the owner's identification and an attribute name, but can also follow a customized specification.

Archetypes must be declared in the `prototype` hierarchy by decorating them with the `archetype` decorator. Then in order to make them usable in practice, a corresponding implementation must be written in the `nxtype` hierarchy. Here is a partial illustrative example:

```python
## AML (partial illustration)

# Defines the Data archetype
@archetype
class Data[T](Object):
    class nx(Object.nx):
        fspec: str = option(default="")  # This is a runtime option
        
        def print(self) -> str: ...  # Methods declared in archetype.nx only hint at the interface

## Implementation (partial illustration)
from .aml import Data

# Generic Data archetype
class DataNx[T](ObjectNx, implements=Data):
    fspec: str = Data.nx.fspec.NX
	_data: T | None  # _data is a private attribute held for optional caching
    
    def __call__(self, *args, **kwargs) -> T:
        return self.data(*args, **kwargs)
    
    def print(self) -> str:
        return f"{self.data():{self.fspec}}"        
        
## User code
import anaximander as nx

# Defines the Float prototype that specializes Data
class Float(nx.Data[float]):
    nx.options.fspec = ".2f"  # This syntax sets the fspec option to ".2f" for the Float type
```

As a result of these statements, the compiled Float type in the DTI behaves as expected:

```python
## DTI
>>> float = Float(3.141)
>>> float.nx.print()
'3.14'
```

Let us now dissect this example:

* `Data[T]` is declared as a generic prototype with type parameter `T`, inheriting from the base prototype for all digital twin representations, which is `Object`.
* Because archetypes and traits encode attributes and behaviors that are accessible from the digital twin interface through the `.nx` attribute, their declarations are nested in an inner class `nx`. The main reason for this is to provide consistency across coding constructs -for instance, the `print` method is referenced as `Float.nx.print` whether `Float` is the AML prototype or the digital twin interface. Note that the inner class `nx` may create a namespace conflict when importing `anaximander` as `nx`. However this is unlikely in practice because a) there is typically little reason for archetypes to declare anything in their class body that is outside of the `nx` inner class[^1], b) conversely, prototypes that are not decorated as archetypes have no use for an `nx` inner class.
* The inner class `nx` declares so-called nxdescriptors, which are applicable to archetypes and traits behind the `.nx` accessor, as well as the methods that interfaces should expose. This last point deserves a bit of an explanation. At runtime in the DTI, `float.nx` evaluates to an `nxobject` instance of the class `FloatNx`, which is compiled from the AML `Float` declaration. Hence the available methods and attributes are driven by the `FloatNx` implementation. However, the Anaximander compiler also creates a stub file for the `Float` interface. This stub file effectively picks the attributes that are made visible to type checkers and IDEs, which serves to eliminate undesirable items such as dunder methods or implementation details. The  mechanics for this is that the compiler will select the attributes and methods that are explicitly declared in the archetype's `nx` inner class. These do not require an implementation -the implementation comes from the matching `nxtype` class.
* `Data.nx` declares two attributes: an nxdescriptor `fspec` and a method `print`. The nxdescriptor is an option. Options are runtime parameters that are declared in archetypes and traits. Concrete prototypes may set their own default value or even freeze the value for all their instances. At runtime, instances may set their own value by passing it to their init method -unless of course the value is frozen by their prototype. 
* `DataNx` is the implementation of the `Data` archetype. This is signaled by importing `Data` from the AML, and passing the `implements` keyword argument in `DataNx`'s class header. Next, the nxdescriptors have to be referenced, hence `fspec: str = Data.fspec.NX`. The `.NX` property of nxdescriptors is a reference that is tailored to the `nxtype` system. 
* The example doesn't show the implementation of `Data.data`, which is complex and not the focus of this example. It does show that the `__call__` method is just a proxy for it. The print method is straightforward, note that it also calls `Data.data` in order to materialize the data value before printing it.
* Finally, the last statement shows how to declare a `Float` data type in AML.  It subclasses `Data` with the type parameter `float`, and it further sets a type-level default value for the `fspec` option. It is important to note that `nx.options.fspec = ".2f"` does not globally mutate the default value of `Data.nx.fspec`. The assignment is interpreted by the prototype metaclass to be a local assignment.

### Traits

Traits complement the archetype system with mixed-in behaviors that can be composed across derived archetypes. Traits are defined much in the same manner as archetypes. The main difference is that, as mix-in classes, traits cannot define structural characteristics. A trait must inherit from an archetype and can then only be applied to prototypes that subclass that archetype. Here is an illustration of the measurement trait that adds a physical unit to a Data type. The convention is to use lower case class names for traits so that they are not confused with archetypes.

```python
## AML

# Defines the measurement trait
@trait
class measurement(Data):
    class nx(Data.nx):
        unit: str = meta()  # This is a metacharacter


# Implementation
from .aml import measurement

class measurementNx(DataNx, implements=measurement):
    unit: str = measurement.nx.unit.NX
    
    def print(self):
        data = super().print()
        return f"{data} {self.unit}""
    
## User code
import anaximander as nx

class VelocityMPH(nx.Float, nx.measurement, unit="mph"):
    pass
```

As can be seen, the basic mechanics of declaring and implementing traits follow the same pattern as for archetypes. Here the `measurement` trait defines the `unit` metacharacter. A metacharacter is a metadata property that participates in type creation. In particular, it becomes a valid keyword argument in prototype definitions, as shown in the `VelocityMPH` declarative header. Note that it is also possible to set metacharacters using the same syntax that was used for the `fspec` option in the previous section, like so:

```python
class VelocityMPH(nx.Float, nx.measurement):
    nx.meta.unit = "mph"  # Assigns the 'unit' metacharacter
```

In practice, most metacharacters are set as type constants by passing a literal, like `unit` here. However it is possible to supply a field or field expression instead by using the `fx` keyword, like:

``` python
class MyRecord(nx.Record, nx.sample):
    nx.meta.freq = nx.fx("frequency")
    ...
    frequency: int = <some lookup function TBD>
```

The above construct assumes that the collection frequency of MyRecord can be looked up based on the collection device identity and time. [NOTE: THIS KIND OF SELECTION SYNTAX IS STILL UNDEFINED]. 

Finally, it is also possible to set metacharacters as type properties. For this pattern, supply a callable written as a class method with a single `cls` argument to the `factory` keyword argument, like:

```python
# Additional details on `Data` declaration
from typing import get_args

@archetype
class Data[T](Object):
    class nx(Object.nx):
        ...
        # Fills the dtype metacharacter by looking up type parameter
        dtype = meta(factory=lambda cls: get_args(cls.prototype)[0])  
```

When a new `nxtype` is created based on the `Data` archetype, the `dtype` metacharacter is evaluated at initialization. Here it introspects the supplied `prototype` and extracts the type parameter, e.g. `Data[int] -> int`. 

Additionally, traits can be lifted to data aggregates. For instance, a table, or dataframe of sample records is expected to feature a column or combination of columns that describes an event timestamp. In order to implement this concept, a `sampleDataTableNx` trait is defined, and indicates that it lifts the `sample` trait. Here is a simplified version that glosses over complex timestamp definition scenarios:

```python
## AML

@traitlift(sample)
class sampleDataTable(DataTable):
    class nx:
        @property
        def timestamp(self) -> Column[datetime]: ...

## Implementation
from .aml import sampleDataTable

class sampleDataTableNx(DataTableNx, implements=sampleDataTable):
    
    @property
    def timestamp(self) -> Column[datetime]:
        attr = self.schema.nxfields["timestamp"]
        return self.table[attr]
```

Trait lifts are generally assigned automatically but in some cases they are susceptible to be used in definitions of representations. For instance, we could imagine assigning a `trajectory` trait to a table of sequential spatial records, which among other features sets a `linestring` property from the locations of individual records:

```python
## AML

@trait
class trajectory(sampleDataTable, locationDataTable):
	class nx:
        @property
        def linestring(self) -> Data[LineString]:...

## Implementation

class trajectoryNx(DataTableNx, sampleDataTableNx, locationDataTableNx, trait=True):

	@property
	def linestring(self) -> Data[LineString]:
		return Data(lambda: self.table.location().tolist())

## User code

class TripLog(DataTable[VehicleRecord], trajectory):
    pass

```

We have now defined a `TripLog` type that specializes `DataTable[VehicleRecord]` with the `trajectory` trait. 

### protodescriptors

We have already seen multiple examples of protodescriptors used to specify model attributes, as well as nx-prefixed attributes of archetypes and traits. This section expands the description of their interface and implementation.

Besides archetype and trait inheritance, protodescriptors define the unique characteristics of prototypes. The bulk of protodescriptors serve to declare model fields. These distinguish between data fields, which are embedded in the model and part of its primary storage medium, and relation fields, which express composition between models, including foreign-key relationships, selection queries and pointers to documents. Additional protodescriptors specify schema characterisitcs, such as field unicity or indexing and partitioning schemes, and data validation methods. More details on these descriptors can be found in the AML design documents. Then there are so-called nxdescriptors, which apply to archetypes and traits and take their name from the fact that they are accessed through the `.nx` attribute of interface objects. Following is a description of these nxdescriptors, namely metacharacters, options and archefields.

#### NxDescriptors

##### Metacharacters

Metacharacters are metadata attributes that affect the behavior of `nxobject` instances. A metacharacter declared in an archetype's or trait's implementation becomes a valid keyword argument for derived classes. Metacharacters are declared with the `meta` function. Examples above included `dtype` in the `Data` archetype, and `unit` in the `measurement` trait.

In prototype declarations, metacharacters may be assigned in the class header, or inside the class body by prefixing the metacharacter name with `nx.meta`. In `nxobject` classes, these metacharacter assignments become class-level, read-only properties.

Note that there is a functional need to create metacharacters whose value is instance-dependent rather than set at the class level. For instance, the `sample` trait specifies a sampling frequency for records. In future versions, this frequency may need to be key and time-dependent. This would be concretely implemented by explicitly passing a field expression to the metacharacter setter, as in this statement: `nx.meta.freq = nx.fx("frequency")`. 

Hence the setter will interpret the supplied positional argument according to three possible cases:

1. A literal assignment, e.g. `"mph"`, which sets the value at the type level
2. A callable -in that case, the callable must operate on an `nxtype` instance, which is to say a `cls` argument, and is called at class creation
3. A field expression, which can wrap either a field reference or a new expression as a callable that takes an interface instance as its argument

##### Options

Options are runtime options that govern behavior like eager loading and caching. They are defined in archetypes and traits. Unlike metacharacters, they can be changed at the instance level. Further, prototypes may declare their own default, and they may also freeze options for their instances.

##### Nxfields

"Nxfields" could also be called logical or semantic fields -one alternative naming option was "archefield". They are not model fields per se, but they either map to a model field or a combination thereof. For instance, a record that implements the `sample` trait must declare an event timestamp. The actual model field that provides this timestamp can take various names or even be made up of multiple component fields (e.g. date and time). Nxfields behave like fields and return DTI objects but they are only accessed on `nxobject` instances, as in `record.nx.timestamp` -hence the "nxfield" naming. They are declared in archetypes or traits using the `nxfield` descriptor function. For example:

```python
@trait
class sample(Model):
    class nx:
        freq: str = nx.meta()
        timestamp: datetime = nx.nxfield()
   
class MyRecord(Record, sample, freq="5T"):
    key: str = nx.field(key=True)
    ts: datetime = nx.field(timestamp=True)
```

The MyRecord class signals that its `ts` field operates as the `timestamp` nxfield.

## Core nxtype System

### nxobject Interface

`nxobject` is the workhorse of the type system. It provides the system methods, particularly data CRUD methods, as well as common data representation and summarization methods. While `nxobject` can cache data, it should be understood as an object-data mapping interface. It gets instantiated from a interface, to which it provides various data manipulation and materialization methods.

`nxobject` is an abstract base class, hence it doesn't get instantiated in practice. Subclasses specialize the interface in various ways but all build on a common core structure that comprises the following members:

class members:

* `prototype: prototype`: - the class' prototype, from which `archetype`, `traits` and `metacharacters` are derived
* `archetype: prototype` - the class' archetype
* `traits: tuple[prototype]` - the class' traits
* `metacharacters: Mapping[str, Any]` - the class' metacharacters (held as dictionary in `__metacharacters__`)
* `__options__: dict[str, Any]` - a mapping of default options for class instances

instance members:

* `dti: Object`: a weak reference to the DTI `Object` instance that the object implements 

* `state: NxState` - the instance's state, which encodes information such as validation, integrity, caching, etc.
* `options: Mapping[str, Any]` - a mapping of options for the instance
* `_options: dict[str, Any]` - instance-specific options supplied at instantiation

### nxtype metaclass

Classes derived from `nxobject` implement the `nxtype` metaclass. To create new classes at runtime, `nxtype` expects an archetype that serves as base class, optional traits that derive from the base class, and metacharacters.

`nxtype` is responsible for the following:

* Validates the combination of supplied archetype, traits and metacharacters
* In the case of archetypes and traits, collects and processes nxdescriptor proxies -accessed as `nxdescriptor.NX`
* Assigns values to nxdescriptors in concrete `nxtype` classes

## Interface Type System

### Object Interface

* `nxtwin: NxTwin` - the digital twin instance that the object is pointing to. Digital twin instances can be deployment instances (e.g. "production", "staging", etc.) and they can also be replay or simulation instances. At any rate, the digital twin instance provides the storage context within which the object is defined. In most cases, the digital twin instance is set globally and individual objects don't need to specify it.

* `nxdef: NxDef` - the instance's specification, which is a reference that uniquely identities a data object. Exact specs are TBD but provisionally, a definition could be:
  * An entity ID
  * A record (key, seq) pair
  * An `NxScope` instance, which specifies a spatiotemporal frame and optionally entity types and identifiers
  * An executable query function (i.e. a view)
  * An attribute, defined as the combination of a string and `NxSpec`
  * A literal (e.g. a number)
  * A callable

## Notes on Compilation



[^1]: In theory one could imagine declaring a base schema and a behavioral interface within a single class decorated by @archetype. If so, placing the inner class `.nx` below other declarations mitigates the problem. However it is still TBD whether such a construct should even be allowed.
