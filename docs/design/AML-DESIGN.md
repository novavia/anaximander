# AML DESIGN

## Overview

### Language Scope

The Anaximander Modeling Language (AML) is the starting point for users of the Anaximander framework. It is exposed as a Python library and allows developers to model a physical world environment by breaking it down into individual concepts -or said more formally, define an ontology. The syntax is declarative for the most part, organized into classes that represent these concepts. At a high level, the language defines base classes for object-based representations as follows:

* Data is used for the definition of elementary data types assigned to model fields. Most digital twin developers may not need to create additional data types beyond those provided by the framework, but it is fully extensible in this regard.
* Model is used for the definition of structured data representations comprised of fields and relationships. AML developers can define arbitrary Models and reference them in other models. The Model base class has no persistence semantics, however. It gets subclassed into three archetypes that do:
  * Entity describes identifiable things that make up the modeled environment - places, vehicles, machines, people, etc. Entities are persistent (i.e. their existence expand beyond a single point in time, and they are identified by a unique id) and stateful (i.e. they can implement states that change over time, tracked by logs), and their non-state attributes are slowly-changing dimensions whose changes are recorded as well to enable time-travel.
  * Record describes informational data that is immaterial - telemetry, metrology, events, transactions, etc. In data warehousing terminology, these are the facts. Records are immutable and identified by a combination of keys (usually entity ids) and optionally sequencing fields, most commonly time-based. Record serves to model both externally sourced data and derivatives computed by transformation and aggregation.
  * Document is a general-purpose class to persist data models with document semantics: they are accessible by path and can be versioned. They serve to store specifications, configurations, or to decompose Entity or Record attributes into separate units of storage. 

In addition to defining data structures, AML also describes the dynamic aspects of a digital twin system — although this is out of scope for the initial release that this version of the document targets. These are specifically broken down into two planes:

* Record types are organized into a data mesh, forming a directed acyclic graph whose edges are transformation operations. AML employs a data asset paradigm, such that the operation that materializes a given Record type is specified in the class that defines that type, with references to the input types required to run the operation.
* Anaximander also enables an event-driven architecture. Digital twin developers use AML to define conditions that trigger events, as well as listener functions that subscribe and respond to these events. Triggers may be defined inside the transformation operations of the data mesh, which interconnects the two planes. Listener functions can also define further triggers. Another triggering modality is a change in an entity's attribute or state. This aspect of the framework may also be extended into a first-class abstraction for physical world process modeling, which would parallel the structural modeling provided by the Data and Model base classes. 	

### Example

In order to make the specifications intelligible, we begin with an example that introduces the AML syntax and some of its essential concepts. In this example, tri-axis acceleration probes collect bursts of vibrations data from motors every 30 seconds. The model covers:

- **Motors** with manufacturer, model, serial number, and technical nameplate data.
- **Monitoring locations** (head or tail of a motor) where vibration probes can be installed.
- **Vibration probes**, modeled polymorphically to allow different hardware types.
- **Vibration samples**, recorded every 30 seconds, including FFT spectra of acceleration and overall RMS velocity in each axis.
- **Summaries** (hourly and daily), which roll up samples into compact statistics such as mean RMS velocity and banded spectral averages.
- **Incidents**, defined as time-bounded sessions with their own PDF report stored in a data lake.

Together, these elements demonstrate AML’s ability to express entities, records and specs, as well as enumerations, field expressions, and document references in a simplified but still realistic industrial use case.

```python
from typing import ClassVar

import anaximander.aml as nx

# Motor entity class
class Motor(nx.Entity):
    # Here we define a composite key with three data fields
    # Note that entities receive a unique id regardless of key definitions
    manufacturer: str = nx.data(key=True)
    model_name: str = nx.data(key=True)
    serial_number: str = nx.data(key=True)
    
    # The nickname can serve as an alternate retrieval key
    # This is indicated by the 'unique' flag. At the same time, the field is nullable.
    # As a result, uniqueness is understood to apply across non-null values only
    nickname: str | None = nx.data(unique=True)
    
    # This next attribute is associated with a Model type
    # Since it is defined as a data field, it will be stored as an embedded JSON object
    nameplate: "MotorNameplate" = nx.data()
    
    # The next field uses nx.datetime, which at runtime leverages the pendulum library
    commissioning_date: nx.datetime = nx.data()
    
    # Monitors is a back reference ('backlink') based on MonitoringLocation defining a link
    # (and hence a foreign key implementation) to Motor.
    monitors: nx.List["MonitoringLocation"] = nx.backlink()
    
    # Finally, we define an access path, which will result in the generation of dedicated
    # storage and retrieval methods based on nickname (key-based access is built-in by default)
    # The __nxschema__ variable is a special container for enumerating schema descriptors.
    # Here the path descriptor takes a template string as argument. This sets a path template
    # for retrieving model instances, and the name is used to refer to that template. Hence
    # a motor instance can be retrieved by nickname in the following way:
    # Motor.nx.retrieve("M1", template="nickname")
    # And further:
    # motor.nx.path() -> "Siemens/1FW3/12345"  # canonical object path
    # motor.nx.path(template="nickname") -> "M1"  # templated object path
    __nxschema__ = [
        nx.path(t"{nickname}", name="nickname")
    ]
    
# A Model class for nameplates. Note closed=False, borrowed from Python's TypedDict
# indicating that model instances may define additional, arbitrary fields
class MotorNameplate(nx.Model, closed=False):
    # The following fields use the Measurement archetype and define a unit as string
    rated_power: nx.Measurement | None = nx.data(unit="kW")
    rated_voltage: nx.Measurement | None = nx.data(unit="V")
    rated_current: nx.Measurement | None = nx.data(unit="A")
    # Of course measurements / units are optional and one can use plain integers as well
    rated_frequency_hz: int | None = nx.data()
    rated_speed_rpm: int | None = nx.data()
    
    # This field defines a format specification default. Note that Python's float could be
    # equivalently used as a type hint
    efficiency: nx.Float | None = nx.data(fspec=".1%")    

# This next class is an enumeration, which follows the Python's Enum conventions
class MotorEnd(nx.Enum):
    HEAD = "head"
    TAIL = "tail"

# This is an entity class representing either the head or tail of a given motor
# where a vibrations probe may be located
class MonitoringLocation(nx.Entity):
    # A link is an entity relationship implemented by a foreign key. Here the descriptor
    # further specifies that motor is part of the model's key, and hence not nullable,
    # and that MonitoringLocation instances' lifecycle are tied to their assigned Motor,
    # that is, motor is deleted, the deletion cascades to the monitoring location.
    motor: Motor = nx.link(key=True, on_delete="cascade")
    
    # The next field uses the MotorEnd enum as its assigned type
    # Note that the default can be either MotorEnd.HEAD or "head"
    # as it gets coerced.
    motor_end: MotorEnd = nx.data(key=True, default="head")
    
    # Here the backlink is assigned a singular entity type or None, which means
    # that the cardinality of the relationship is 0:1 -this is mirrored by
    # VibrationProbe.location's unique constraint.
    probe: "VibrationProbe" | None = nx.backlink()

# Next is an entity class representing vibration probes
class VibrationProbe(nx.Entity):
    # This class is polymorphic. It defines type-level fields by using ClassVar,
    # and these must be assigned a literal value for concrete types to be generated.
    # The first field also happens to be a typekey, which makes it a type discriminator
    # that uniquely defines an inherited type. It also enables
    # writing VibrationProbe["Acme"] to point to AcmeProbe in the
    # digital twin interface.
    model: ClassVar[str] = nx.data(typekey=True)
    max_frequency: ClassVar[int] = nx.data()
    
    # The link to MonitoringLocation is nullable, which implies
    # that probes can be decommissioned and/or reassigned.
    # Further, the unique flag implies a 0:1 cardinality (at most one
    # probe per location)
    location: MonitoringLocation | None = nx.link(unique=True)

# Concrete vibration probes follow by setting the values of type fields
class AcmeProbe(VibrationProbe):
    model = "Acme"
    max_frequency = 800

class BetaProbe(VibrationProbe):
    model = "Beta"
    max_frequency = 1600

# Next we define specialized measurement types by specifying unit and formatting precision
# The Measurement archetype implements a 'metacharacter' named 'unit'. This can be set
# in a subclass header as done for Acceleration. The archetype also implements the
# 'fspec' runtime option. Unlike metacharacters, which are set for each concrete type,
# options can be supplied for individual instances at runtime, but types can still
# set a default value, as is done here. Further, the option could be frozen by using
# the `Final` keyword from the typing library.
# The syntax for setting options leverages a special interface nx.option where options
# get globally registered -though of course the metaclass will make sure that the assignment
# makes sense. That same syntax can also be used for metadata.
class Acceleration(nx.Measurement, unit="m/s^2"):
    nx.option.fspec = ".3f"
    
class Velocity(nx.Measurement):
    nx.meta.unit = "mm/s"
    nx.option.fspec = ".3f"

# The following model falls in the records category. Records are like facts
# in data warehousing terminology. They are indexed by a key and/or sequence.
# Further, records are specialized according to their time indexing scheme.
# The following record model is of the sample type, as indicated by the sample
# trait in its inheritance chain: it is indexed by key and timestamp,
# and the timestamps are expected to be periodic, or at least approximately so,
# such that they form a dense set -at least locally, as there could be gaps in the index.
class VibrationSample(nx.Record, nx.sample):
    # Sample records expect a freq metacharacter that sets expected frequency
    nx.meta.freq = "30s"  # The data collection frequency
    monitoring_location: MonitoringLocation = nx.link(key=True)
    
    # The next field is marked as the timestamp that is used to sequence records
    # Internally, timestamp (as written inside the descriptor assignment) is
    # a so-called `nxfield`, i.e. a semantic field that is defined from concrete fields.
    # Here the concrete field happens to also be named timestamp, but it could be
    # something else, or date and time could be split between two fields and reconciled
    # as a field expression. Nxfields are accessed in the .nx namespace, hence here:
    # vibration_sample.timestamp -> reference to the concrete model field, specific to this model
    # vibration_sample.nx.timestamp -> reference to the nxfield, which is implemented on all sample models
    timestamp: nx.datetime = nx.data(timestamp=True)
    shaft_speed_hz: int = nx.data()
    probe: VibrationProbe = nx.link()

    # The fx descriptor/decorator stands for 'field expression'. These are lowerable expressions
    # limited to simple operations on fields and other field expressions. Here it
    # specifies the length of one-sided Fourier transform bins. Field expressions are converted to
    # instance properties in the digital twin interface.
    sample_rate_hz: int = nx.data()
    sample_duration: nx.timedelta = nx.data(default="1s")
    fft_length: int = nx.fx(lambda s: int(s.sample_rate_hz * s.sample_duration.total_seconds()) // 2 + 1)

    # The next fields are arrays of acceleration measurements. Their length
    # is specified in the field definition. This is done in two different ways
    # to demonstrate the flexibility of the language. One is to directly reference
    # an already defined descriptor (length=fft_length). The other syntax uses
    # fx again, but with its positional argument set to a string. This signals a
    # field reference rather than a field expression, and can be used for forward references.
    accel_fft_x: nx.Array[Acceleration] = nx.data(length=fft_length)
    accel_fft_y: nx.Array[Acceleration] = nx.data(length=fft_length)
    accel_fft_z: nx.Array[Acceleration] = nx.data(length=nx.fx("fft_length"))
    velocity_rms_x: Velocity = nx.data()
    velocity_rms_y: Velocity = nx.data()
    velocity_rms_z: Velocity = nx.data()

    # Enabling selection by probe as alternative to monitoring location
    # The specification also indicates a unicity constraint on probe, timestamp
    # pairs. Given that probe itself is nullable, the unicity constraint is
    # only applied to non-null values.
    __nxschema__ = [
        nx.unique(probe, timestamp),
        nx.path(t"{probe}/{timestamp}", name="probe")
    ]
    
# Here is another Model that is used embedded in the VibrationSummary model,
# avoiding repetition
class VibrationStats(nx.Model):
    # Root-mean square velocity statistics
    velocity_rms_mean: Velocity = nx.data()
    velocity_rms_stdev: Velocity = nx.data()
    
    # 16 fixed Hz bands (averaged) over the acceleration spectrum (illustrative)
    mean_accel_power_spectral_density: nx.Array[float] = nx.data(length=16)

# VibrationSummary is a journal record. Journals are strictly periodic records
# that are emitted for preset time periods, hence appropriate for summarizations.
# We could have written the inheritance chain as (nx.Record, nx.journal). This
# accounts for the fact that journal is a trait that can be applied to Models, not
# just Record-type models. However, because that is still by far the most common
# configuration, there is a Journal archetype that is basically a redefinition of
# the Record archetype with the journal trait applied to it.
class VibrationSummary(nx.Journal):
    monitoring_location: MonitoringLocation = nx.link(key=True)
    # Journals use a single timestamp as sequencing index, but the
    # timestamp semantically refers to a time period, as set by
    # the freq metacharacter. The next field is designated as the
    # period field.
    timestamp: nx.datetime = nx.data(period=True)
    x_stats: VibrationStats = nx.data()
    y_stats: VibrationStats = nx.data()
    z_stats: VibrationStats = nx.data()
    
    # The logs attribute is described by a window selection over the VibrationSample
    # record collection. The selection uses the summary's monitoring location and
    # the built-in timespan property (from the journal trait) that returns a time interval.
    # The assigned type is a DataSequence of records, that is, a dataframe expecting
    # a single value for the records' key field, leaving the sequence field as the
    # meaningful index.
    logs: nx.DataSequence[VibrationSample] = nx.selection(key=monitoring_location,
                                                          time=nx.nxfield.period)

# Two concrete embodiments of the VibrationSummary base model follow,
# characterized by hourly and daily periodicity, respectively.
class VibrationHourlySummary(VibrationSummary):
    nx.meta.freq = "H"

class VibrationDailySummary(VibrationSummary):
    nx.meta.freq = "D"

# Yet another record type is the session, which sets arbitrary but non-overlapping
# time intervals for a given key. This is used to capture session windows.
class VibrationIncident(nx.Session):
    monitoring_location: MonitoringLocation = nx.link(key=True)
    
    # Sessions require two fields marked as start_time and end_time, respectively
    start_time: nx.datetime = nx.data(start_time=True)
    end_time: nx.datetime = nx.data(end_time=True)
    
    # Another data sequence selection, using the same construct as VibrationSummary's
    logs: nx.DataSequence[VibrationSample] = nx.selection(key=monitoring_location,
                                                          time=nx.nxfield.timespan)
    
    # Here we use the document descriptor, which points to a stored data type such as
    # media or a large matrix, or a model that is stored detached from its parent model.
    # Typically, these would be stored in a data lake or document database, though this need not be
    # the case -other database engines can be used.
    report: nx.PDF = nx.document(path=t"reports/{monitoring_location}/{start_time:%Y-%m-%dT%H-%M-%S}")
```

### Functions and Design Goals

AML declarations are compiled by the Anaximander framework in order to generate the code base of a digital twin system. That code base includes system-level code - database schemas, API endpoint definitions, cloud functions, etc., as well as the Anaximander Digital Twin Interface (DTI). In particular, the DTI transforms the classes declared with AML into classes of the same names, but outfitted with a full set of methods that provide access to the digital twin system and data. This method of code generation is preferred over traditional inheritance and metaprogramming because it provides full transparency, easier integration with a development environment, and the ability to edit the generated code in order to customize behavior beyond the baked-in modeling constructs.

Accordingly, we set the following goals and constraints for AML:

* Usability: AML is the interface that digital twin developers use to specify their system and its evolution. It is of paramount importance to make that interface user-friendly, which includes a clear and straightforward syntax, complete documentation, adequate validation routines with clear warning and error messages, and good integration with the development environment and generative AI tools.
* Transparency: the automated compilation and re-compilation of AML into multiple target libraries means that digital twin developers don't have full control over the code base. Hence automation must be balanced with transparent and deterministic rules so that an acceptable level of control can be maintained. This is to be achieved with the same general strategies described to ensure usability, extended to the compiler packages that generate code.
* Easy is easy, complex is feasible: generally, typical structural and behavioral patterns should be quick and easy to read or write in AML. Conversely, the language must make it possible to implement more complex patterns. All in all, this means that the language must strive to fully decompose and expand all the concepts that it exposes, and overlay syntactic shortcuts that are applicable to simple situations.

## Type System

A more complete description of the Anaximander type system is available in the document of the same name. At a high level, the framework creates three parallel class hierarchies:

* Data models are declared in AML by writing data classes that are instances of the `prototype` metaclass. Prototypes are abstract classes with no other function than to encapsulate modeling declarations that can be compiled or referenced at runtime.

* At the core of the framework, we find the `nxobject` base class and the corresponding `nxtype` base metaclass. This is where the methods of the digital twin interface are written. This class hierarchy defines the abstract interfaces that mediate between the concrete interfaces and system code.
* The high-level interface to the digital twin is realized with concrete interfaces that inherit from the base `Interface` class.

This document focuses on the `prototype` hierarchy.

### Prototypes

All prototypes are abstract classes, which means that they cannot be instantiated. Prototypes are generated by the `prototype` metaclass. In the current design all prototypes are structural data representation prototypes inheriting from the base `Object` class. In the future, it is possible that `prototype` will be employed to spin off a base `Process` class to model physical world actions and transformations, but that is still TBD.

The primary function of the `prototype` metaclass is to read protodescriptors that are declared in the class' body, and verify their coherence. It also uses the standard library `ast` module to collect an abstract syntax tree of the class' declarations, which is leveraged in compilation logic.

As should become clear in the remainder of this section on the type system, a prototype encapsulates three layers of information:

* A single archetype that sets the data structure of the prototype (e.g. Data, Model...), and maps to an implementation class
* Optional traits, which are mix-in only prototype classes that add behavioral features to the prototype
* Metacharacters, which one can think of as class-level keyword arguments, and that impact the runtime behavior of the prototype's interface. For instance, the physical unit of a measurement factors into representation methods implemented by the interface. A much more obvious and rich metacharacter is a model's schema -though the AML interface does not formally identify the schema as a metacharacter and builds it implicitly from field declarations.

These three layers of information are also what the `nxtype` metaclass expects to generate or retrieve a class, which provides the interface's implementation.

### Protodescriptors

In AML, prototype fields are defined with a class of objects named "protodescriptors". This terminology reflects the fact that these objects are eventually compiled into Python descriptors, but that they do not themselves implement the descriptor protocol — hence they are descriptor precursors, or protodecriptors. Protodescriptors form their own class hierarchy. At the highest level, we find the following classes:

* `FieldDescriptor` - field descriptors are the most common descriptor. They serve to describe data fields, compositional relationships and field expressions.
* `MethodDescriptor` - method descriptors take the form of method decorators. Their primary use in the current version is for declaration of data validation methods.
* `SchemaDescriptor` - schema descriptors are a special type of descriptors that qualify a model schema -including unique constraints, indexes and more.

Additionally, there is a class of descriptors that are reserved to archetypes and traits. These are called metadescriptors, and they are accessed through the `.nx` attribute of the `Interface` class, e.g. `vehicle.nx.uuid` points to the unique identifier of a vehicle model instance -here `uuid` is an archetypical field that is not part of the `Vehicle` domain, hence the suffix. `MetaDescriptor` itself is abstract and is the parent to three subclasses:

* `MetaCharacter` - a protodescriptor class for metacharacters, which are type-level parameters that factor into archetype and trait behavior.
* `OptionDescriptor` - options are runtime parameters that generally can be supplied at the object instance level -though a prototype can set a default or even freeze an option's value for all of interface instances to use.
* `NxFieldDescriptor` - so-called nxfields are fields implemented by archetypes and traits. These are generally derived from the prototype's concrete fields or field expressions (`uuid`, presented above as an example, figures as an exception here), and carry semantic meaning. A prime example is the `timestamp` field in a sample or event record model -the actual field may be called something else in the concrete model, but can also be accessed as `record.nx.timestamp` across all record models for consistency.

Besides, there are mixin classes that control protodescriptors behavior:

* `AnnotatableDescriptor` instances are declared with a type assignment.
* `IdentifiableDescriptor` inherits from `AnnotatableDescriptor` and adds the `unique` flag to indicate unicity
* `AssignableDescriptor` inherits from `IdentifiableDescriptor` and presumes that the descriptor may be assigned a value. As a result it adds the `default` and `factory` attributes that govern default assignments.
* Callable descriptors wrap a method. They can be declared either in-line or by using a decorator.
* `FieldListDescriptor` references a list of model fields.

Protodescriptor declarations are wrapped in lower-case functions that provide a user-friendly syntax as seen in the above examples. The classes described below are complemented by corresponding functions, which are explicitly listed in [section 2.4](#protodescriptors-design ). For instance `DataDescriptor` instances are declared with the `data` function.

### Archetypes

Anaximander defines an original type system that is foundational to its modeling language, and the notion of archetype stands as its central pillar. An archetype is in many respects just like an abstract base class. However, unlike in a traditional class hierarchy, archetypes are designed to supply behavior by composition rather than inheritance. This is done in order to separate domain modeling from systems implementation, which is a core goal of the Anaximander framework. 

Archetypes specify the meta-structure of objects, such as whether the underlying data is a single value, a key-value map, a collection, or an entire dataframe — and in the latter case, the indexing scheme. In AML, developers declare prototypes and associate them with an archetype. This is done by using Python's built-in inheritance mechanism. Hence a prototype always inherits from a single parent archetype -and can optionally inherit from traits. Following is a basic example:

```python
import anaximander as nx

# Defines a data type based on the Measurement archetype
class Temperature(nx.Measurement):
    nx.meta.unit = "Celsius"

# At runtime, Machine is a prototype tied to the Entity archetype
class Machine(nx.Entity):  
    name: str = nx.data(key=True)   
    kind: str = nx.data()
    commission_date: nx.Date = nx.data()
```

When `Machine` is compiled into an interface class, it exposes the `.nx` attribute, which evaluates to an `nxtype` -or an `nxobject` instance of that type, if the attribute is suffixed from a Machine instance. The `nxtype` class inherits from `EntityNx`, which is the implementation of the `Entity` archetype. In this way, the `Machine` domain is kept separate from the system functions that the framework implements.

Archetypes serve to define data types and models, but they can also express derived, aggregate structures:

```python
# In turn, classes based on aggregate archetypes can be defined from the same prototypes, e.g.
nx.List[Temperature]  # Class that holds a list of temperature measurements

class MachineDataFrame(nx.DataFrame[Machine]):  # Class that exposes a dataframe containing machine definitions
    
    @nx.metric
    def mean_age(self) -> nx.Duration:
        today = nx.today().date()
        age = self.commison_date.apply(lambda d: (today - d).days)
        return nx.duration(days=age.mean())
```

In the above example, `MachineDataFrame` is a declarative prototype that adds a custom metric to the composition of the generic `DataFrame` archetype and the `Machine` prototype.

Developers can also define their own archetypes by decorating an archetype subclass with the archetype decorator. The decorated class is first interpreted as a prototype, and the decorator introspects its contents to establish it as a new archetype. However, archetype declarations look significantly different from those of regular prototypes. In order to establish consistency with the interface implementation, archetypes (and traits) declare their attributes and methods in an inner `nx` class:

```python
# The decorator establishes DataTile as a new archetype derived from the Record archetype
@nx.archetype
class DataTile(nx.Record):
    cell_index: str = nx.data()  # cell_index is a regular data field, part of the schema of subclasses
    
    class nx(nx.Record.nx):
        zoom_level: int = nx.meta()  # zoom_level is a metacharacter, set by concrete types as a class variable
```

### Traits

Traits enhance the Anaximander type system with mix-in classes that encapsulate bags of functionality and can be reused alongside multiple archetypes. They effectively reduce the number of necessary archetype definitions by factoring out combinations of behaviors. For instance, if a model prototype defines a geometry field, then the generated concrete type as well as its various aggregates implement the `geometric` trait, which adds properties and methods expected of geometric objects. 

```python
class Street(nx.Entity, nx.geometric):
    ...
    geom: nx.LineString = nx.field(geometry=True, nullable=False)
    ...
```

In this particular case, the model needs to feature exactly one field that is marked as the `geometry` nxfield. Note that this doesn't prevent other fields from having geometric types, but only one field can be marked as the model's primary geometry.

A trait inherits from an archetype and can only be passed to prototypes that also inherit from that archetype. This architecture is paralleled in the `nxtype` implementation hierarchy: the trait's implementation inherits from the archetype's implementation. Further, traits can get lifted to aggregate types. For instance, the `geometric` trait is a mix-in class for model types, and is lifted to `geometricDataFrame` mix-in class for dataframe types that are composed with a geometric model. However this is generally transparent to end users of the framework.

In the same way that archetypes are defined with the `nx.archetype` decorator, the mix-in trait classes are defined by decorating them the `nx.trait` decorator. And just like with archetypes, the type system that is offered out-of-the-box by the framework can be extended and customized by end users. 

### Built-in Functions

The framework offers a set of built-in, specialized functions accessed through dedicated interfaces for temporal, math, string, and geometric operations. These functions can be used both in AML or at runtime, but their primary intent is the former, where they serve as symbolic expressions that can be compiled to the various backend libraries.

#### `nx.dt` — Temporal Functions

- `nx.dt.now()` – current timestamp
- `nx.dt.today()` – current date
- `nx.dt.add(x, **kwargs)` – add duration to temporal value
- `nx.dt.truncate(x, unit)` – floor to unit (`"day"`, `"hour"`, etc.)
- `nx.dt.start_of(x, unit)` – start-of period
- `nx.dt.end_of(x, unit)` – end-of period
- `nx.dt.year(x)` / `month` / `day` / `hour` / … (field extractors)
- `nx.dt.interval(start, end)` – construct temporal interval
- `nx.dt.contains(interval, value)`

#### `nx.math` — Numeric Functions

Core portable operations:

- `nx.math.abs(x)`
- `nx.math.sqrt(x)`
- `nx.math.exp(x)`
- `nx.math.log(x)`
- `nx.math.floor(x)` / `ceil(x)` / `round(x)`
- `nx.math.pow(x, y)`

Mapped to SQL / Ibis / Arrow equivalents and to Python’s math module.

#### `nx.str` — String Functions

- `nx.str.lower(x)`
- `nx.str.upper(x)`
- `nx.str.length(x)`
- `nx.str.substr(x, start, length)`
- `nx.str.trim(x)`
- `nx.str.replace(x, old, new)`
- `nx.str.concat(x, y, …)` (optional)

#### `nx.geo` — Geometry Functions

- `nx.geo.contains(a, b)`
- `nx.geo.intersects(a, b)`
- `nx.geo.distance(a, b)`
- `nx.geo.buffer(x, r)`
- `nx.geo.area(x)`
- `nx.geo.length(x)`

## Object Archetypes and Traits

The current version of Anaximander focuses on structural data representations, and this section describes the concrete object archetypes that ship with the framework, as well as the built-in traits that are compatible with them. The top-level hierarchy unfolds as follows:

```yaml
Object				# Parent archetype to structural data representations
├── Data            # Parent archetype for elementary data types
├── Struct          # Base archetype for key-value mappings
|   ├── Model   	# Parent archetype for model types
|   └── Folder		# Archetype for mappings of documents and folders
├── Media           # Parent archetype for media data (images, audio, documents...)
├── Aggregate       # Parent archetype for aggregate types
└── View       		# Archetype for views, which can expose arbitarily shaped data queries
```

* Data is the base archetype for data types. Objects that implement the Data archetype expose a single attribute, held as `._data`.
* Struct is a base archetype used to hold key-value maps.
  * Model is the base archetype for all model types.
  * Folder is a special archetype designed to emulate structured storage. Folders are defined as key-value maps that only accept references to documents or other folders.
* Media is the base archetype for media objects, including audiovisual content and publications.
* Aggregate is the base archetype for data representations that combine multiple data or model items. They particularly include dataframes and collections, and support property lifting.
* View is the archetype used to encapsulate arbitrary queries, for which the modeler may not necessarily want to specify the expected structure. As a general rule, materialized views that participate in a data mesh should be declared with a model archetype, but views that are only used as reports are easier to declare with the View archetype.

The `Object` archetype implements the following members:

```python
@archetype
class Object[T](Arche):
    class nx[Proto: "Object[T]"](Arche.nx[Proto]):
        dtype: type[T] = nx.meta()  # Metacharacter for the evaluation type of the prototype's interface
        path: Path = nx.nxfield()  # Nxfield that provides the access path to an Object instance
        
        def __call__(self, **kwargs) -> T:...
        	"""Materializes an instance's data structure."""
```

Cutting across all object archetypes is a trait related to data persistence, titled `stored`, though it is itself abstract. The first concrete descendent is `artifact`, which assumes the simplest form of object storage and retrieval, such as a document in a file system or an entry in a key-value store.

```yaml
Object
└── stored			# Parent trait for stored objects
    └── artifact    # Trait for document-like stored objects
```

The other concrete descendants of the `stored` trait are specializations for entities and records, which are described along their respective archetypes.

```python
@trait
class stored[T](Object[T]):
    class nx[Proto: "Object[T]"](Object.nx[Proto]):
        store: Store = nx.meta() # An object store instance, which points to a storage table or bucket
        id: int | str | UUID = nx.nxfield() # A unique identifier for the object relative to its store
        
        @classmethod
        def create(cls: type[Proto], **attrs, **meta, **options) -> Proto:...
        	"""Creates and stores an object instance."""
        
        @classmethod
        def bulk_create(cls: type[Proto], *objects, **meta, **options) -> "Set[Proto]":...
        	"""Bulk store object creation."""
                
        @classmethod
        def get(cls: type[Proto], id: int | str | UUID, **meta, **options) -> Proto:...
        	"""Retrieves a single instance based on id or raises."""
        
        def exists(self) -> bool:...
        	"""True if the calling interface can be identified in storage."""
        
        @classmethod
        def retrieve(cls: type[Proto], *args, template: str | None = None, **meta, **options) -> Proto | None:...
        	"""Retrieves a single instance or None based on a supplied path, which may be passed as a chain or literal path."""
        
        @classmethod
        def select(cls: type[Proto], **kwargs, **meta, **options) -> "Table[Proto]":...
        	"""Selects any number of instances based on basic filters (exact syntax TBD)."""
        
        def correct(self, **attrs) -> Proto:...
        	"""Corrects an object's stored representation as of the current system time."""
        
        @classmethod
        def bulk_correct(cls: type[Proto], **corrections: dict[Proto, dict]) -> "Set[Proto]":...
            """Bulk instance correction."""
        
        def delete(self) -> Proto:...
        	"""Deletes an instance from store."""
        
        @classmethod
        def bulk_delete(cls: type[Proto], *instances: Proto) -> "Set[Proto]":...
        	"""Bulk-deletes instances from store."""


@trait
class artifact[T](stored[T]):
    class nx[Proto: "Object[T]"](stored.nx[Proto]):
        def update(self, *, as_of: datetime | None, **attrs) -> Proto:...
        	"""Updates an artifact as of now or the current event time."""
        
        @classmethod
        def bulk_update(cls: type[Proto], *, as_of: datetime | None, **updates: dict[Proto, dict]) -> "Set[Proto]":....
        	"""Bulk instance update."""
```

### Data Archetypes

`Data` is the simplest archetype. It exposes a single data value, though that data can be of arbitrary complexity. Anaximander offers archetypes for scalar data, compound data (i.e. tuples), and arrays of various dimensions and shapes.

```python
from decimal import Decimal
from enum import Enum
from uuid import UUID
from typing import Literal, TypeVar

type PyScalar = bool | bytes | Decimal | float | int | IntEnum | Literal | str | StrEnum | UUID

@archetype
class Data[T](Object[T]):
    class nx[Proto: "Data[T]"](Object.nx[Proto]):
        dtype: type[T] = nx.meta()  # Metacharacter for the evaluation type of the prototype's interface
        fspec: str | None = nx.meta()  # An optional metacharacter for a format specification
        def data(self, **kwargs) -> T:...
        	"""Archetype-specific materialization method."""
            
@archetype
class Scalar[T: PyScalar](Data[T]):
    class nx[Proto: "Scalar[T]"](Data.nx[Proto]):
        dtype: type[T] = nx.meta()
        
# Notes: T: PyScalar may need to be expanded to include Data prototypes themselves
# In previous version, we had written:
# type PyData = PyScalar | tuple[PyScalar]
# type DataFieldType = nx.Data | nx.Document | nx.Collection[DataFieldType]
# where nx.Collection[T] is nx.List[T] | nx.Set[T] | nx.Dict[K, T]
# all immutable, where K = TypeVar("K", int, str)
```

The `Data` archetype also registers two important base traits:

* The `measurement` trait adds physical unit metadata to a `Data` prototype. As a shortcut, the `Measurement` archetype is a derivation from `Data[float]` with the `measurement` trait added.
* The `category` trait signals that the domain of a `Data` prototype is limited to discrete values. `category` is further subclassed into `enum`, `lookup` and `hierarchy`. However in the vast majority of cases, categories use string representations, and the `Enum`, `Lookup` and `Hierarchy` archetypes provide reference archetypes for that situation.
* The `observation` trait confers temporal, and optionally key and spatial metadata to a `Data` object. It is provisioned in order to transfer these properties from a `Record` to one of its attributes. The `observation` trait could possibly be used to define time series of individual measurements.

Accordingly, the data archetypes are organized according to the following diagram:

```yaml
Data					# Parent archetype for elementary data types
├── measurement			# Trait that add physical unit
├── category			# Trait for categorical data
|   ├── enum			# Trait for enumerations
|   ├── lookup			# Trait for categories referenced by external table
|   └── hierarchy		# Trait for hierarchical categories
├── observation			# Trait for observation data, which carries frame metadata
├── Scalar				# Archetype for scalar data types
|   ├── Measurement	    # Archetype for float physical data measurements
|   ├── Temporal	    # Parent archetype for temporal data types (Date, Time, DateTime, Duration)
|   └── Category	    # Parent archetype for string-based categorical types
|       ├── Enum		# Archetype for enumerated categorical types
|       ├── Lookup		# Archetype for categorical types that reference external tables
|       └── Hierarchy	# Archetype for hierarchical categories
├── Compound			# Archetype for compound data types -i.e. tuple-like structured types
|   ├── Interval	    # Generic archetype for intervals of scalar types
|   ├── MultiInterval   # Generic archetype for sets of disjoint intervals
|   └── Geometry	    # Parent archetype for spatial types (Point, LineString, etc)
└── Array				# Parent archetype for array-based data types
    ├── Vector			# Archetype for 1-dimensional arrays
    ├── Matrix			# Archetype for 2-dimensional arrays
    └── Tensor			# Archetype for n-dimensional arrays
```

Here are some additional notes on the intent and design of these archetypes:

* **Scalar**: the `Scalar` archetype is a base type for framework or user-defined scalar data types, including dedicated implementations of integer, float and other elementary data types.
  * **Measurement**: the `Measurement` archetype is an important concept in the context of digital twins, where data often represents physical phenomena. As of this writing, the plan is to add physical dimension and unit, alternatively level of measurement, as well as accuracy metadata to a scalar float.
  * **Temporal**: parent archetype for the `Time`, `Date`, `DateTime`, and `Duration` data types.
  * **Category**: this parent archetype defines data types that are restricted to discrete strings values.
    * **Enum**: this is the simplest form of category, which is defined by enumerating a possibly ordered set of discrete, acceptable values.
    * **Lookup**: the lookup `archetype` functions as an enumeration whose admissible values is defined outside of code, presumably in a stored or external data source. This can be useful for large and/or standardized enumerations such as time zones, spatial reference systems, or stock-keeping units.
    * **Hierarchy**: the `Hierarchy` archetype takes categorization one step further by allowing admissible values to be defined in a tree-like manner. This can particularly useful for describing states -for instance, a machine asset can be either in commission or out of commission. If it is in commission, it can be either operating or idle. As such, the operating and idle state are hierarchical children of the in-commission state.
  
* **Compound**: the `Compound` archetype is the basis for data types that are treated as a single value but are represented by tuples of scalars. Some good examples are geometric points or RGB-defined colors. This archetype particularly branches into the following archetypes:
  * **Interval**: archetype for single-axis intervals based on a scalar types
  * **MultiInterval**: archetype for sets of disjointed intervals for a given scalar type.
  * **Geometry**: the base type for the six basic types of the GIS stack (`Point`, `LineString`, `Polygon`, `MultiPoint`, `MultiLineString`, `MultiPolygon`)

* **Array**: the `Array` archetype is a base archetype for arbitrarily large, homogeneous numerical data structures.
  * **Vector**: a one-dimensional array
  * **Matrix**: a two-dimensional array
  * **Tensor**: an n-dimensional array

### Struct Archetypes

The `Struct` archetype is an abstract parent archetype for key-value maps, which particularly include data models.

#### Model Archetypes

All structured data representations in AML derive from the `Model` archetype. A `Model` defines a schema of fields and can be validated, serialized, and nested. Stored model archetypes extend `Model` to express how structured data objects exist and evolve within a system.

As already described in the [Overview](#Overview) section, there are four base model archetypes:

```yaml
Model   		# Parent archetype for model types, and archetype for value-only models
├── Entity   	# Archetype for persistent entities
├── Record   	# Archetype for informational data
└── Document   	# Archetype for models stored in documents
```

* **Model**: the `Model` archetype can be used to define arbitrary data models and embed them as submodels in other model definitions. They can also be reemployed as field blocks. Hence it is possible to define models in a modular fashion and bring together multiple model types by coalescing their fields.
* **Entity**: entities are persistent and uniquely identifiable objects. The framework automatically assigns them a unique identifier, and it is therefore not required to define a key, but of course it is still recommended since the identifier is arbitrary. Entity fields can implement states. State attributes are assumed to change somewhat frequently and their values are stored as time series rather than alongside the non-state fields, which are considered slowly-changing dimensions.
* **Record**: records must be indexed by a key field or field tuple, and optionally a sequence. The most common combination is a key alongside a timestamp, but a static lookup set can be defined by a `Record` class with a single key field. Records cannot define states and their field values are immutable -they can still be amended to make up for input errors, but this is a system operation rather than a functional operation. Records can also define relations to other models.
* **Document**: the `Document` archetype serves to persist models as documents. `Document` models do not define keys or sequence fields.

Here is a summary of key features and differences:

| Archetype    | Persistence pattern             | Identity          | Temporal semantics                    | Typical backend          | Examples                                                     |
| ------------ | ------------------------------- | ----------------- | ------------------------------------- | ------------------------ | ------------------------------------------------------------ |
| **Model**    | Not stored or embedded as value | None              | None                                  | None or database field   | Configuration, Specification                                 |
| **Entity**   | Stored *by reference*           | Key / ID          | Stateful (slowly changing dimensions) | Relational / Graph       | Asset, Device, User                                          |
| **Record**   | Stored *by append*              | (Key, sequence)   | Immutable (event stream)              | Time-series / Fact table | Observation, Log entry, Measurement                          |
| **Document** | Stored *by value*               | Path / URI / hash | Versioned (immutable revisions)       | Object / Document store  | Configuration, Specification, Report, Detached Record Subset |

`Model` provides the common abstraction for any structured type. It declares named fields, type annotations, and optional validation or transformation logic. A `Model` may be used transiently (e.g. as a DTO or Pydantic-style schema) or serve as the base for one of the storage archetypes below.

```python
class Calibration(nx.Model):
    parameters: dict[str, float] = nx.data()
    author: str = nx.data()
    created_at: datetime = nx.data()
```

An **Entity** represents a durable object with a distinct identity and evolving state. Entities are stored *by reference*: they have a stable key (or ID) and are updated in place. In general, entity attributes are slowly-changing dimensions (SCD) which may be logged to enable time-travel. Entities also define explicit states, which are presumed to change dynamically and frequently and are hence persisted as records. Entities themselves are stored in a relational or graph database for persistence.

```python
class Device(nx.Entity):
    serial_number: str = nx.data(key=True)
    firmware_version: str = nx.data()
    calibration_date: datetime = nx.data()
```

*Entity Characteristics*

- Persistent and addressable via ID or key.
- Support updates (event-time modifications), corrections (system-time modifications), deletions, and versioned state histories.
- Participate in relationships and compositions.
- Typical compiler targets: ORM classes, SQL tables, graph nodes.

A **Record** captures an immutable observation or event, generally associated with an `Entity`. Records are stored *by append* and are identified by a composite key, often including a timestamp. They model streams, logs, and time-series measurements.

```python
class TemperatureSample(nx.Record, nx.sample):
    device: Device = nx.link(key=True)
    timestamp: datetime = nx.data(timestamp=True)
    temperature_c: float = nx.data()
```

*Record Characteristics*

- Immutable after creation, can be corrected and versioned.
- Represent facts in time (not mutable state).
- Support partitioning and time-based queries.
- Typical compiler targets: fact tables, append-only logs, time-series databases.

A **Document** represents a structured value object stored in a file-like system or a document database. Unlike an `Entity`, a `Document` has no intrinsic identity or state. Documents are immutable once versioned. Each revision yields a new document instance rather than mutating the existing one.

*Document Characteristics*

- Stored by value; addressable via path or content digest.
- Versioned but not stateful.
- Ideal for specifications, configurations, and nested structured data.
- Typical compiler targets: JSON or Parquet documents, object-store blobs, document database collections.

Models can implement the following **traits**:

<u>**Temporal Traits**</u>

The temporal traits enable specialization for models that define event-time attributes. There are six temporal traits as follows:

* **sample**: the sample trait attaches to models that define an event timestamp and an expected frequency. This trait is primarily applicable to records that are collected periodically or pseudo-periodically, such as pulling readings from field sensors. While there can be time gaps in the collection, the records are densely indexed.
* **event**: the event trait attaches to models that define an event timestamp. By contrast with samples, events are expected to be sparsely indexed and are appropriate for recording times when a specific condition is met.
* **transition**: the transition trait specializes the event trait to mark a transition between two states. Models that implement the transition trait must mark one of their fields as the state field, which by convention represents the state that is initiated at the time of the event timestamp.
* **journal**: the journal trait attaches to models that define data summarization over fixed time windows -such as hourly or daily summary, which may be tumbling (disjoint) or hopping (overlapping). The time windows are indexed with an event timestamp, but the model semantics assume that this timestamp represents a window.
* **session**: the session trait attaches to models that define both a start and end times. This trait is applicable to records of session windows, in which the windows for a given key do not overlap.
* **phase**: the phase trait specializes the session trait for sessions that span between two state transitions -and hence a phase corresponds to a single steady state. Like with transitions, models that implement the phase trait must mark one of their fields as the state field.

**<u>Spatial Traits</u>**

In the initial implementation of Anaximander, there are two spatial traits:

* **located**: the located trait attaches to models that define a single location. This is appropriate for telemetry records on moving entities, such as vehicles or robots.
* **geometric**: the geometric trait attaches to models that define a geometry field. The primary intended use case is for entities whose spatial footprint is captured by the data model.

It is typical for the sample and location traits to be combined. The combination can be specified readily as **waypoint**. This trait gets lifted in aggregate models, either as **itinerary** such that the sequence of locations defined by consecutive records resolves to a MultiPoint geometry, or a **trajectory**, in which case the sequence resolves to a LineString geometry.

#### Folder Archetype

The `Folder` archetype serves as an interface to collections of documents held in hierarchical storage. It is not necessary to use folders to store documents: a document is accessed through a path that can be arbitrarily nested, and directories are created as needed. Defining a folder is effectively a way to create an interface that provides directory-level operations. `Folder` falls under the `Struct` archetype as it defines arbitrary field names. However these fields are restricted to the `document` and `folder` descriptors only. 

### Media Archetypes

The `Media` parent archetype is tailored to BLOB content. It is specifically subclassed as follows:

```yaml
Media				# Parent archetype for media types
├─ PDF      		# pdf
├─ Image 			# Raster or vector
├─ Audio  			# wav, mp3 
└─ Video         	# mp4, avi
```

### Aggregate Archetypes

Aggregate archetypes are primarily useful as part of Anaximander's Digital Twin Interface in order to manipulate query results that are comprised of multiple models. However, they also feature in AML as type hints, particularly for specifying the return type of selection descriptors, and in future versions for the inputs and outputs of data transformation operators. But it is also possible to declare prototypes using an aggregate archetype: this can be useful to create metrics for aggregates of a particular model type. For instance:

```python
class VehicleLog(DataSequence[VehicleRecord]):
    
	@metric
    def max_speed(self) -> Velocity:
        return Velocity(self.speed.max())       
```

The `Aggregate` hierarchy unfolds as follows:

```yaml
Aggregate       		# Parent archetype for aggregate types
├── Series        		# Archetype for series of data objects (Scalar, Measurement, Category, Compound)
| 	├── SerialSequence
| 	├── SerialMapping
|   └── SerialLog
├── DataFrame       	# Archetype for tables of model objects, most commonly records
| 	├── DataSequence
| 	├── DataMapping
|   └── DataLog
└── Collection      	# Archetype for collections of objects
	├── List
	├── Set
    └── Dict
```

* **Series**: the `Series` archetype can be composed with types that are derived from the `Data` archetype to represent data series. This archetype is further specialized to account for specific indexing schemes.
  * **SerialSequence**: a sequentially-indexed series
  * **SerialMapping**: a key-indexed series
  * **SerialLog**: a series that is indexed by both key and sequence -in which case data ordering is relative to a given key value
* **DataFrame**: the `DataFrame` archetype can be composed with model types to create table-like structures whose columns are field values. This archetype is further specialized to account for specific indexing schemes.
  * **DataSequence**: a sequentially-indexed dataframe
  * **DataMapping**: a key-indexed dataframe
  * **DataLog**: a dataframe that is indexed by both key and sequence -in which case row ordering is relative to a given key value
* **Collection**: this is a base archetype for typical object collections. The advantage of declaring collection prototypes over built-in Python collection types is that the archetypes are composed with the element type to generate a runtime class that is equipped with attributes, properties and methods derived from the element type -and recursively so as applicable. For instance, `List[Temperature]` is unit-aware and supports conversions. However this is obviously only applicable to homogeneous collections.
  * **List**: an ordered object sequence
  * **Set**: an unordered set of objects without duplicates
  * **Dict**: a key-value mapping of objects

### View Archetype

The `View` archetype serves to hold views, which is to say, the result of arbitrary queries.

## Protodescriptors

Here is the fully expanded protodescriptor hierarchy, indicating for each class:

* Whether it is abstract, a mixin class, and what mixins it inherits from in addition to its parent
* And second, the corresponding declarative AML function -which in the case of callable descriptors may also be used as a method decorator

```yaml
Protodescriptor
├── AnnotatableDescriptor				(mixin)
│   └── IdentifiableDescriptor      (mixin)
│       └── AssignableDescriptor    (mixin)
├── CallableDescriptor              (mixin)
├── FieldListDescriptor             (mixin)
├── MetaDescriptor                    (abstract)
│   ├── MetaCharacter               ← mixin: AssignableDescriptor        							shortcut: meta
│   ├── OptionDescriptor            ← mixin: AssignableDescriptor        							shortcut: option
│   └── NxFieldDescriptor           ← mixin: AssignableDescriptor        							shortcut: nxfield
├── FieldDescriptor                 (abstract)
│   ├── DataDescriptor              ← mixin: AssignableDescriptor, IdentifiableDescriptor			shortcut: data
│   ├── RelationDescriptor          ← mixin: AssignableDescriptor
│   │   ├── LinkDescriptor          ← mixin: DataDescriptor                 						shortcut: link
│   │   ├── BackLinkDescriptor      ← mixin: DataDescriptor                 						shortcut: backlink
│   │   ├── SelectionDescriptor     ← mixin: CallableDescriptor, IdentifiableDescriptor             shortcut: selection
│   │   ├── ViewDescriptor          ← mixin: CallableDescriptor             						shortcut: view
│   │   ├── DocumentDescriptor      ← mixin: DataDescriptor                 						shortcut: document
│   │   ├── FolderDescriptor        ← mixin: DataDescriptor                 						shortcut: folder
│   │   └── StateDescriptor         ← mixin: CallableDescriptor, IdentifiableDescriptor             shortcut: state
│   ├── FieldExpressionDescriptor   ← mixin: CallableDescriptor, IdentifiableDescriptor   			shortcut: fx
│   ├── FieldGroupDescriptor        ← mixin: AssignableDescriptor, FieldListDescriptor				shortcut: fieldgroup
│   ├── FieldBlockDescriptor        ← mixin: IdentifiableDescriptor                					shortcut: fieldblock
│   └── MetricDescriptor            ← mixin: AssignableDescriptor, IdentifiableDescriptor 			shortcut: metric
├── MethodDescriptor                (abstract) ← mixin: CallableDescriptor, AssignableDescriptor
│   └── ConstructionDescriptor      (abstract)
│       ├── ParserDescriptor                                           								shortcut: parser
│       └── ValidatorDescriptor                                        								shortcut: validator
└── SchemaDescriptor
    ├── KeyDescriptor               ← mixin: FieldListDescriptor       								shortcut: key
    ├── SequenceDescriptor          ← mixin: FieldListDescriptor       								shortcut: sequence
    ├── UnicityDescriptor           ← mixin: FieldListDescriptor       								shortcut: unique
    ├── IndexDescriptor             ← mixin: FieldListDescriptor       								shortcut: index
    ├── PartitioningDescriptor      							       								shortcut: partition
    ├── PathDescriptor                                              								shortcut: path
    └── SortDescriptor                                              								shortcut: sort
```

Following is a short description for each subclass:

- **Protodescriptor**: the abstract base for all declarative model descriptors in AML, defining the common interface and shared metadata for attributes declared in class bodies.
- **AnnotatableDescriptor (mixin)**: adds annotation awareness, allowing descriptors to interpret type hints in field declarations.
  - **IdentifiableDescriptor (mixin)**: adds the unique flag for descriptors that point to identifiable objects or metadata.
    - **AssignableDescriptor (mixin)**: provides assignment semantics, including support for defaults and factories, as well as validation.
- **CallableDescriptor (mixin)**: wraps callable objects or functions, enabling function-like descriptors such as computed views or expressions.
- **FieldListDescriptor (mixin)**: references an ordered list of model fields in a `fields` attribute.
- **MetaDescriptor (abstract)**: base class for members of the `.nx` internal interface, defining metadata and configuration accessors.
  - **MetaCharacter** ← mixin: `AssignableDescriptor`: declares type-level metadata (“metacharacters”) configuring model-wide properties such as units or serialization.
  - **OptionDescriptor** ← mixin: `AssignableDescriptor`: defines configurable runtime options.
  - **NxFieldDescriptor** ← mixin: `AssignableDescriptor`: defines archetypical fields that are used in archetype behavior implementation.
- **FieldDescriptor (abstract)**: base for instance-level attributes representing model fields or data members.
  - **DataDescriptor** ← mixin: `AssignableDescriptor`, `IdentifiableDescriptor`: describes a data-bearing field linked to persistence or serialization.
  - **RelationDescriptor (abstract)** ← mixin: `AssignableDescriptor`: defines inter-model relationships referencing other entities or collections.
    - **LinkDescriptor** ← mixin: `DataDescriptor`: defines a single-valued reference (foreign-key–like) to another entity.
    - **BackLinkDescriptor** ← mixin: `DataDescriptor`: represents the inverse side of a link relationship.
    - **SelectionDescriptor** ← mixin: `CallableDescriptor`, `IdentifiableDescriptor`: declares a parameterized selector or query over related objects.
    - **DocumentDescriptor** ← mixin: `DataDescriptor`: references external documents or media artifacts stored by path.
    - **FolderDescriptor** ← mixin: `DataDescriptor`: represents a collection or directory of documents or media.
    - **StateDescriptor** ← mixin: `IdentifiableDescriptor`: defines a stateful attribute for entities, which is looked up in a state log.
  - **FieldExpressionDescriptor** ← mixin: `CallableDescriptor`, `IdentifiableDescriptor`: represents a computed field derived from an expression or function.
  - **FieldGroupDescriptor** ← mixin: `AssignableDescriptor`, `FieldListDescriptor`: groups multiple field definitions into reusable bundles.
  - **FieldBlockDescriptor** ← mixin: `IdentifiableDescriptor`: coalesces a model's fields to the declaring model, and makes it a group.
  - **MetricDescriptor** ← mixin: `AssignableDescriptor`, `IdentifiableDescriptor`: defines aggregate metrics on data or model aggregates.
- **MethodDescriptor (abstract)** ← mixin: `CallableDescriptor`, `AssignableDescriptor`: declares callable methods embedded in models for processing or validation.
  - **ConstructionDescriptor (abstract)** ← mixin: `CallableDescriptor`, `AssignableDescriptor`: groups parsing, normalization, and validation methods.
    - **ParserDescriptor**: parses raw or external inputs into structured model instances.
    - **ValidatorDescriptor**: defines validation logic for fields or objects.
- **SchemaDescriptor (abstract)**: defines schema-level elements governing structure, identity, and data organization.
  - **KeyDescriptor** ← mixin: `FieldListDescriptor`: declares one or more fields as the primary identity key.
  - **SequenceDescriptor** ← mixin: `FieldListDescriptor`: defines sequential or versioning order.
  - **UnicityDescriptor** ← mixin: `FieldListDescriptor`: enforces uniqueness constraints across combinations of fields.
  - **IndexDescriptor** ← mixin: `FieldListDescriptor`: defines secondary indexing strategies for lookup or optimization.
  - **PartitioningDescriptor**: specifies partitioning logic for physical or logical data segmentation.
  - **PathDescriptor**: maps instances to routing templates that define them uniquely.
  - **SortDescriptor**: defines the model’s default sorting or ordering behavior.

The following subsections highlight the design features of these classes.

### Protodescriptor

The `Protodescriptor` base defines the minimal metadata common to all AML descriptor types. This section describes protodescriptor behavior and attributes, including the admissible declaring prototypes, the admissible assignable types, and the admissible types of protodescriptor attributes.

Certain attributes use the `MISSING` singleton sentinel as their default value to indicate that they are unset. The type of `MISSING` is `Missing` by convention, and in the framework's implementation.

**Attributes:**

- `name: str | None` — the name of the descriptor. This is an internal attribute that is not settable through declarative functions. In the vast majority of cases, the name is set by a declarative attribute.

- `owner: prototype` — the prototype that declares the descriptor.

- `doc: str | None`  — short human-readable documentation.

- `config: Config`  — arbitrary configuration parameters. These can be expressed in a dictionary with string keys, or by passing `DescriptorConfig` objects, per the provisional definition below:

  ```python
  type ConfigValue = Any | DescriptorConfig | Mapping[str, ConfigValue]
  type Config = DescriptorConfig | Mapping[str, ConfigValue]
  ```

  For the initial release, configs will simply be a placeholder for arbitrary key-value mappings of extra attributes. In time, we will implement normalized, reusable configuration objects to simplify the developer experience, with the following provisional rules:

  - **Normalization:** resolve any `DescriptorConfig` via `resolve() → Mapping[str, Any]`, then deep-merge into a single mapping.

  - **Inheritance merge:** parent → child deep-merge; child overrides scalars/type mismatches; dicts deep-merge; lists concatenate.

  - **Templating (optional):** `${name}` and `${owner}` placeholders in string values are resolved **after** merge/normalize.

  - **Deletion:** support a sentinel (e.g., `nx.DELETE`) to remove inherited keys.


### Mix-in Descriptors

Mixin descriptors are abstract base classes that contribute cross-cutting semantics to concrete protodescriptors. They are not declared directly in AML syntax but are composed into other descriptor types to enrich their behavior.

| Class                        | Parent                   | Attributes (name: type)                                      |
| ---------------------------- | ------------------------ | ------------------------------------------------------------ |
| **`AnnotatableDescriptor`**    | `Protodescriptor`        | `annotation: str`, `type: prototype | metatype`, `nullable: bool` |
| **`IdentifiableDescriptor`** | `AnnotatableDescriptor`    | `unique: bool | None = None`                                 |
| **`AssignableDescriptor`**   | `IdentifiableDescriptor` | `default: Any = MISSING` , <br />`factory: Callable[[], Any] | Missing = MISSING`, <br />`parser: Callable | Iterable[Callable] | None = None`, <br />`validator: Callable | Iterable[Callable] |None = None`, <br />`gt, ge, lt, le, min_length, max_length, pattern` |
| **`CallableDescriptor`**     | `Protodescriptor`        | `callable: Callable[..., Any]`                               |
| **`FieldListDescriptor`**    | `Protodescriptor`        | `fields: tuple[FieldDescriptor, ...]`                        |

Together, these mixins define the reusable behavioral units from which all higher-level protodescriptors are composed.

- **Attributes:**
  - `annotation: str` — the annotation string input in AML. This is an internal attribute that is not settable through declarative functions.
  - `type: prototype` — the resolved admissible prototype for the descriptor, or a metatype in the case of `MetaCharacter` and `Option`. If an abstract base class uses a type union, the closest common ancestor prototype (resp. metatype) is assigned. This is an internal attribute that is not settable through declarative functions.
  - `nullable: bool`  — whether the descriptor is nullable, which is derived from its type hint. This is an internal attribute that is not settable through declarative functions.
  - `unique: bool | None = None`  — whether the field's value is unique across instances. If the field also happens to be nullable, the constraint applies across non null values. If `None`, the behavior is dictated by the descriptor's other features. For instance a single key field must be unique and that does not need to specified. In most cases a `None` unique resolves to `False`.
  - `default: Any = MISSING`  — a default value.
  - `factory: Callable[[], Any] | Missing = MISSING`  — a default factory.
  - `parser: Callable | Iterable[Callable] | None = None`  — optional transformation function(s) applied to the assigned value before validation. A single callable or an iterable of callables may be supplied; the framework normalizes this to an ordered list of parser functions.
  - `validator: Callable | Iterable[Callable] | None = None`  — optional predicate function(s) applied to the assigned value after parsing. A single callable or an iterable of callables may be supplied; the framework normalizes this to an ordered list of validator functions.
  - `gt, ge, lt, le, min_length, max_length, pattern`  — in addition to custom parsers and validators, assignable descriptors may declare inline constraints analogous to Pydantic’s field constraints.
  - `callable: Callable[..., Any]`  — for callable descriptors, this is the callable that they wrap, and it is a positional attribute.
  - `fields: tuple[FieldDescriptor, ...]`  — for field list descriptors, the normalized tuple of referenced field descriptors.

#### Assignable types

In general, for descriptors whose values fall within the digital twin domain, the assignable types that can be associated with the protodescriptor can be either prototypes or types that can be inferred to prototypes. The framework makes these type definitions in order to make the code clean.

At the top of the representation prototypes hierarchy we find `Object`. As a result, we define `ObjectType` as any type that either inherits from `Object` or that can be interpreted as a prototype. To give a concrete example, `ScalarType` would include:

```python
from decimal import Decimal
from enum import Enum
from uuid import UUID
from typing import Literal, TypeVar

type PyScalar = bool | bytes | Decimal | float | int | IntEnum | Literal | str | StrEnum | UUID

ScalarType = PyScalar | Scalar[ScalarType]
```

The same notation (`<Archetype_name>Type`) is used without further explanations through the remainder of the document to refer to an equivalent pattern.

Apart from using `| None`, type unions are not allowed except in abstract classes, assuming that the descriptor is redefined with a single type. 

#### Redefinition and reassignment rules

Redefinition refers to setting a new protodescriptor definition with the same name as a protodescriptor defined in a parent prototype. By contrast, reassignment refers to the situation in which a protodescriptor's value has been set in a prototype (that's only applicable to type-level descriptors), and a derived prototype makes an assignment to the same attribute.

For most protodescriptors, redefinition in child classes is simply not allowed, but there are a few exceptions. Likewise, for type-level descriptors that are set in prototypes, children classes may generally not set new attributes, but again there are exceptions.

* **MetaCharacter**: Redefinition is allowed in derived traits and archetypes only if they tighten the admissible type or constraints. If a base prototype assigns a value to a metacharacter, then a derived prototype may only reassign that value if the value itself can be tightened. For instance if a metacharacter takes a class as its assigned value, then the derived prototype may reassign with a class that inherits from the parent's metacharacter value.

* **NxFieldDescriptor**: Redefinition is allowed in derived traits and archetypes only if they tighten the admissible type or constraints. No reassignment is permitted on derived prototypes.

* **OptionDescriptor**: Redefinition is allowed in derived traits and archetypes only if they tighten the admissible type or constraints. Prototypes can set their own default values irrespective of parent's assignments. If a prototype freezes an option, then no reassignment is allowed either in derived prototypes or in interface instances.

* **FieldDescriptor**: Redefinition is allowed in derived prototypes if it is a monotone tightening. If a field descriptor is defined as a `ClassVar` and it is assigned a value in a derived prototype, then a further derived prototype may not reassign. Tightening lattice:

  - Nullability: `True → False`

  - Required: `False → True`

  - Cardinality (min..max): increase minimum and/or decrease maximum (e.g., `0..* → 1..* → 1..1`)

  - Type: narrow to a subtype or smaller domain (e.g., `str → Literal[...]`, `float → PositiveFloat`, `DF[schema A] → DF[schema A′⊆A]`)

  - Loading behavior: `lazy → eager` (monotone tightening). Cache policy refinements (if introduced later) must only become *stricter*.

  - Cascade / on_delete: more restrictive policies only

### MetaDescriptor

`MetaDescriptor` is the abstract base for members of the `.nx` inner interface. These descriptors model type-level, framework-internal affordances that are exposed under `.nx` , i.e. metacharacters, nxfields and options.

- Scope: prototype-level, though options may be supplied at the instance level at runtime unless explicitly frozen
- Declaration: realized through AML functions (shortcuts) below; they may be read/used by archetype code paths.

Here is a summary of their definition:

| Class                 | Kind     | Inherits mixins        | AML function (shortcut) | **Admissible owner types** | **Admissible assigned types** |
| --------------------- | -------- | ---------------------- | ----------------------- | -------------------------- | ----------------------------- |
| **MetaDescriptor**    | abstract | —                      | —                       | —                          | —                             |
| **MetaCharacter**     | concrete | `AssignableDescriptor` | `meta`                  | `nx inner class`           | `MetadataType`                |
| **OptionDescriptor**  | concrete | `AssignableDescriptor` | `option`                | `nx inner class`           | `MetadataType`                |
| **NxFieldDescriptor** | concrete | `AssignableDescriptor` | `nxfield`               | `nx inner class`           | `ObjectType`                  |

Note the difference in the admissible assigned types. Metacharacters and options accept metadata types. The exact shape of `MetadataType` is still TBD but it will combine standard Python types and dedicated types for metadata -for instance, physical units, or data permissions. By contrast, nxfields are anchored in the digital twin domain, and may only be assigned domain types.

The MetaDescriptor functions (`meta`, `option`, and `nxfield`) also play a double syntactic role as the declarative interface and an assignment interface. This is best illustrated by an example:

```python
import anaximander as nx

@nx.archetype
class Foo[T](nx.Object[T]):
    class nx[Proto: "MyArchetype[T]"](Object.nx[Proto]):
        bar = nx.meta()

class Baz(Foo):
    nx.meta.bar = "baz"      
```

As can be seen the `bar` metacharacter is declared with `bar = nx.meta()`, and the concrete prototype `Baz` sets a value for it with `nx.meta.bar = "baz"`. In practice, these functions are a special class of callable objects acting as declarator handles. nxdescriptors are registered as attributes of that handle, and these attributes have a special setter that uses the current context to make a targeted assignment whose scope is limited to the class in which the assignment takes place, here `Baz`.

### FieldDescriptor

`FieldDescriptor` is the abstract base for instance-level attributes declared in prototypes. It controls how values are typed, loaded, serialized, and represented.

- **Attributes:**
  - `load: Literal["eager", "lazy"] | None = None` — default runtime loading behavior for this field.
  - `repr: bool | Callable | str | None = None` — whether the field appears in basic, default object representations. If a callable is supplied, then the callable is applied to the attribute's value to format it. If a string is supplied, it is interpreted as a format specification. The default is `None`, which delegates the behavior to the calling descriptor type -for instance, a key field will participate in instance representation by default, whereas an arbitrary data field does not.
- **Validation rules**:
  - If the `repr` value implies that the field must be loaded for representation, then `load` cannot be `lazy`.


#### DataDescriptor  ← mixins: AssignableDescriptor, IdentifiableDescriptor (AML: `data`)

Data descriptors are the most straightforward descriptors. They are only applicable to Model prototypes and target assignable attributes of model instances, excluding foreign keys, which are captured by link descriptors. Data descriptors are instances of `FieldDescriptor` but are created with the helper function `data` as so:

```python
class MyModel(nx.Model):
	my_field: int = nx.data()
```

Data descriptors must specify a type hint, and the following types are admissible:

- `DataType` includes all `Data` archetypes as well as the underlying dtypes compatible with it.

- `ModelType` includes prototypes whose archetype is `Model`. However this is a strict requirement: models derived from `Entity`, `Record` or `Document` are not admissible type hints for `DataDescriptor`. If a data descriptor is assigned a model type, the model's serial form is stored with its owner as a submodel.

- `Collection[DataType]` means homogeneous lists/tuples/sets/dicts (dict keys = `int | str`).

  * Built-in Python collection types, limited to `list`, `tuple`, `dict` and `set` containing a single type of data elements. `dict` keys are restricted to integers or strings.

  * Anaximander collection types, specifically `List`, `Dict` and `Set`, again containing a single type of data elements. `Dict` keys are restricted to integers or strings.

The `data` function takes the following keyword-only arguments:

| **Argument** | **Type**      | **Default**      | **Description**                                              |
| ------------ | ------------- | ---------------- | ------------------------------------------------------------ |
| `index`      | `bool | None` | `None` (`False`) | Whether the field should be indexed.                         |
| `required`   | `bool | None` | `None` (`False`) | Models support partial loading, except for required fields   |
| `typekey`    | `bool | None` | `None` (`False`) | Flags the field as a polymorphic discriminator               |
| `key`        | `bool |None`  | `None` (`False`) | Flags the field as key or part of the key                    |
| `sequence`   | `bool | None` | `None` (`False`) | Flags the field as sequence or part of the sequence          |
| `timestamp`  | `bool |None`  | `None` (`False`) | Flags the field as event timestamp (applicable to samples and events) |
| `start_time` | `bool |None`  | `None` (`False`) | Flags the field as event start time (applicable to sessions and phases) |
| `end_time`   | `bool |None`  | `None` (`False`) | Flags the field as event end time (applicable to sessions and phases) |
| `period`     | `bool |None`  | `None` (`False`) | Flags the field as event period (applicable to journals)     |
| `location`   | `bool |None`  | `None` (`False`) | Flags the field as a geometric or geophysical location       |
| `geom`       | `bool |None`  | `None` (`False`) | Flags the field as a geometry attribute of the model         |

**Conventions & exclusions**

- **Tri‑state booleans** (`None`/`True`/`False`):
  `None` ⇒ framework infers behavior from other flags and type; absent any provision, it behaves as indicated in the default's parentheses.
  `True` ⇒ asserted on; `False` ⇒ asserted off.
- **`required=True`** means the field must be present when materializing a model instance (no partial load omission). It is independent of nullability.
- If a field is part of **`key`** or **`sequence`**, it is **implicitly non‑nullable**: including `None` in the type hint will raise an error.
- **Implied indexing**: fields flagged as `key`, `sequence`, or **temporal** flags are **always indexed**. If a field is the **sole** key, it is also **unique**.
- **`index=True`** ⇒ an index is emitted even if not implied by `key/sequence`. Indexing strategy is deferred to compilers based on field type, but may expand to admit options in future versions. Indexes may also be defined at the schema level (see schema descriptors below).
- **Temporal flag exclusivity**:  `timestamp`, `start_time`, `end_time`, `period` are **mutually exclusive** with each other and with `sequence`. It is also not allowed to define multiple such fields (save for `start_time` and `end_time`, which must always be paired).
- **Type compatibility**:
  - `timestamp` / `start_time` / `end_time` / `period` can only be `True` if the assigned descriptor type is time‑like.
  - `location` can only be `True` if the assigned descriptor type is a geometry/geo type; compilers may emit spatial indexes where supported.
- Constraint emission (unique, partial‑unique on nullable, spatial indexes) follows **target‑engine best practices**; exact DDL may vary by backend while preserving semantics.
- Inferred defaults from `None` values are **stable and deterministic** given the rest of the field’s flags and the model’s role (e.g., entity vs record).
- Some of the attributes are shortcuts for schema descriptors (described below). Conflicting or ambiguous declarations across these two paradigms will be handled on a case by case basis using the following principles:
  - If conflicts are detectable, they will generally raise an exception
  - As a general rule, schema descriptors are more verbose but also more expressive and precise, and they will prevail in ambiguous cases

#### LinkDescriptor(RelationDescriptor)  ← mixin: IdentifiableDescriptor (AML: `link`)

`LinkDescriptor` declares a single-reference relationship from a Model to an Entity (i.e., it materializes a foreign key from the owner model to a target entity). As with other annotatatable descriptors, nullability derives from the type hint (`T | None` ⇒ FK nullable). Indexing of foreign-key columns is implied; backends may add composite indexes consistent with best practices. Polymorphic links are allowed if the type hint resolves to an entity supertype; compilers enforce that actual targets are compatible at runtime and may add discriminator/typekey support where required.

**Admissible owner:** `Model`
**Admissible assigned types:** `EntityType`

`LinkDescriptor` accepts the following attributes, in addition to those it inherits:

| **Argument** | **Type**                              | **Default**      | **Description**                                              |
| ------------ | ------------------------------------- | ---------------- | ------------------------------------------------------------ |
| `on_delete`  | `"restrict" | "set_null" | "cascade"` | `"restrict"`     | FK delete action: **restrict** (disallow target deletion), **set_null** (set FK to NULL; requires nullable), **cascade** (delete owner rows/objects with target). |
| `key`        | `bool | None`                         | `None` (`False`) | Whether this link participates in the **owner’s key**. If `True`, the FK columns become part of the model key; implies non-nullable and unique semantics consistent with the model’s key definition. |

<u>Notes</u>:

- `set_null` is invalid if the link is non-nullable.
- `cascade` is the composition-like posture; use it **deliberately** (lifecycle coupling).
- If `key=True`, conflicting nullability is rejected.

As a result of these semantics, the following data modeling patterns may be implemented:

- **Parent/Owner/Composition**: express by combining `nullable=False` with `on_delete="cascade"`. This couples lifecycles (composition-like).
- **Aggregation/Association**: use `nullable=True` and `on_delete="restrict"` by default; `set_null` can be used for soft decoupling.
- **One-to-one**: `unique=True` on the owner side (or make it `key=True` when the target identity is part of the owner identity).
- **One-to-many**: model **on the many side** (child has the singular `link` to parent). 
- **Many-to-many**: use an **association model** by using backlink (see below).

#### BackLinkDescriptor(RelationDescriptor)  ← mixin: IdentifiableDescriptor (AML: `backlink`)

Declares the **reverse traversal** of one or more forward `link`s into an **Entity**. It is a virtual, read-only relation (no FK columns are created); used for navigation and projections. The type hint may be singular (one-to-one reverse) or a collection (one-to-many, or many-to-many via `via`).

**Admissible owner:** `Entity`
**Admissible assigned types:** `ModelType | Collection[ModelType]`

`BackLinkDescriptor` accepts the following attributes, in addition to inherited ones:

| Argument | Type         | Default | Description                                                  |
| -------- | ------------ | ------- | ------------------------------------------------------------ |
| `via`    | `ModelType`  | `None`  | Optional **association/through** model used to traverse many-to-many. If omitted, the backlink targets direct forward `link`s pointing to `owner`. |
| `limit`  | `int | None` | `None`  | An optional limit on the number of collection items that are materialized by default. |

Notes:

- Cardinality follows the type hint: a singular hint requires the forward side to be unique (or key-based) toward the owning model.
- Use `via=AssociationModel` where that model has two `link`s: one to owning model, one to the peer entity.
- Inheritance may only reduce `limit`.

#### SelectionDescriptor(RelationDescriptor)  ← mixin: IdentifiableDescriptor (AML: `selection`)

`SelectionDescriptor` declares a named subset of related objects, defined by a selection query. The query itself may be specified in different ways:

* Using explicit keyword attributes that define a selection frame, i.e. a key or list thereof, a temporal component (datetime or interval), and optionally a filter, e.g.

  ```python
  class UserSession(nx.Session):
      ...
      logs: nx.DataSequence[UserLogs] = nx.selection(key=user, time=nx.nxfield.timespan) 
  ```

* Using a field expression (see below). The field expression may reference other model's attributes but must abide by the limitations of field expressions. One of the advantages of this form is that it can be pushed down to query engines.

  ```python
  class User(nx.Entity):
  	...
  	last_session: UserSession = nx.selection(limit=1)
      last_logs: nx.DataSequence[UserLogs] = nx.selection(fx=lambda s: s.last_session.logs)
  ```

* Using a SQL statement. In this case, the SQL statement is executed at runtime after binding parameters. It may optionally be parametric with respect to event-time arguments.

  ```python
  class User(nx.Entity):
  	...
      order_history = nx.selection(sql=lambda s: t"select * from orders where user.id={s.nx.id}")
  ```

* Using a callable form, either in-line or decorated, that returns an ibis expression. The callable may optionally accept event-time arguments. 

  ```python
  class UserSession(nx.Session):
  	...
  	@nx.selection(kind="ibis")
      def clicks(self) -> nx.DataSequence[UserLogs]:
          T = UserLogs.nx.table
          return (
              T.filter(
                  (T.user_id == self.nx.id)
                  & (T.action == "click")
                  & (T.timestamp >= self.start_time)
                  & (T.timestamp < self.end_time)
              )
          	.order_by(T.timestamp)
      	)
  ```

Irrespective of the query specification, selections come in two flavors: static or dynamic. Static selections are invariant with respect to event time, as are all of the above examples. Note that static selections may still be prone to returning different results over time, since entities' attributes are slowly-changing dimensions, but the semantics and implementation that govern this kind of evolution are different from those put in place for dynamic selections. Dynamic selections are explicitly tied to event time. For callable forms, this means that the event time must be supplied as an argument, optionally with a default. The event time may either be a single value, or a value and a lag, or lastly a value and a rank. If the selection is specified with a field expression, then the only way in which the selection is dynamic is if a member of the field expression is itself dynamic, and no additional event-time semantics need to be provided. For the parametric form, the event-time binding is provided in the the time argument. Here are a couple of examples:

```python
# Dynamic selection examples
class Vehicle(nx.Entity):
    ...
    telemetry_logs: nx.DataSequence[VehicleTelemetry] = nx.selection(time=(nx.dt.lag("5T"), nx.dt.now())) # Defaults to last 5m
    
    @nx.selection(kind="ibis")
    def speed_events(self, time=nx.dt.now(), lag=nx.dt.lag("M")) -> nx.DataSequence[VehicleTelemetry]:
        T = VehicleTelemetry.nx.table
        return (
            T.filter(
                (T.vehicle_id == self.nx.id)
                & (T.overspeed == True)
                & (T.timestamp >= lag)
                & (T.timestamp < time)
            )
        	.order_by(T.timestamp)
    	)
```

**Admissible owner:** `Model`
**Admissible assigned types:** `ObjectType` (singular or collection)

`SelectionDescriptor` accepts the following attributes, in addition to inherited ones:

| Argument   | Type                                                         | Default | Description                                                  |
| ---------- | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
| `kind`     | `"ibis" | "sql" | None`                                      | `None`  | When used as decorator, specifies the kind of callable being wrapped. |
| `callable` | `Callable | None`                                            | `None`  | When used as decorator, the callable being decorated.        |
| `sql`      | `Callable | None`                                            | `None`  | Used for in-line sql form declaration (equivalent to `callable` with `kind="sql"`) |
| `ibis`     | `Callable | None`                                            | `None`  | Used for in-line ibis form declaration (equivalent to `callable` with `kind="ibis"`) |
| `fx`       | `Callable | None`                                            | `None`  | Specifies a field expression                                 |
| `source`   | `type[Record] | None`                                        | `None`  | Specifies a record collection as the source for parametric selections. This is generally optional if the typing annotation already makes that obvious, but it may be necessary in case the selection operates a projection that modifies the result type. |
| `key`      | `Any | list[Any] | None`                                     | `None`  | Discrete key constraint on the selected model. A single key or a list/set of keys. Composite keys may be supplied as tuples or structured values. |
| `time`     | `datetime | tuple[datetime, datetime] | list[tuple[datetime, datetime]] | None` | `None`  | Temporal constraint defining a time point, interval, or set of intervals within which objects are selected. |
| `space`    | `Geometry | tuple[Geometry, Geometry] | list[tuple[Geometry, Geometry]] | None` | `None`  | Spatial constraint defining a location, bounding box, or set of regions used for spatial selection. |
| `filter`   | `Callable | Expr | None`                                     | `None`  | Boolean condition applied to restrict the selection.         |
| `sort`     | `str | list[str] | None`                                     | `None`  | Default ordering for the selected objects. Prefix with `-` for descending. |
| `limit`    | `int | None`                                                 | `None`  | Upper bound on the number of elements materialized by default. |

<u>Notes</u>:

- `kind`, `func`, `sql`, `ibis`, `fx` and other attributes exclude one another. Pick one declarative form only.
- Range attributes (`key`, `time`, `space`) can be singular values, closed/open intervals, or multi-interval collections.
- Filters, ranges, and sorting are combined conjunctively; order of application is implementation-defined but deterministic.
- `time` and `space` may activate index-based optimizations when supported by the backend.
- Inheritance may only narrow filters or ranges, or reduce `limit`.

- The selection's callable receives the owner instance and returns an object or iterable consistent with the declared type, if any.
- Compilers may translate simple view callables to native SQL views, computed columns, or API endpoints, depending on backend.
- Inheritance rules follow standard tightening semantics; overriding a view with the same name replaces its callable.

#### DocumentDescriptor(RelationDescriptor)  ← mixin: IdentifiableDescriptor  (AML: `document`)

Declares a reference to a document/artifact addressed by a path/URI and optionally typed by format/compression hints, accepting the following attributes:

| Argument      | Type                                | Default | Description                                                  |
| ------------- | ----------------------------------- | ------- | ------------------------------------------------------------ |
| `path`        | `str | FieldExpression[str] | None` | `None`  | Locator or owner-derived locator template; opaque identifier, not required to carry extension. |
| `format`      | `str | None`                        | `None`  | Representation hint (enum or MIME). Guides reader/serializer and validation; not required if inferrable. |
| `compression` | `str | None`                        | `None`  | Compression hint when not inferrable from the locator.       |

#### FolderDescriptor(RelationDescriptor)  ← mixin: IdentifiableDescriptor  (AML: `folder`)

Declares a reference to a folder-like container—a namespace for documents, subfolders, or artifacts. The descriptor stores a **locator** (path or URI), not the folder contents.

**Admissible owner:** `Model`
**Admissible assigned types:** `FolderType`

`FolderDescriptor` accepts the following attributes, in addition to inherited ones:

| Argument | Type                     | Default | Description                                                  |
| -------- | ------------------------ | ------- | ------------------------------------------------------------ |
| `path`   | `str | Expr[str] | None` | `None`  | Locator or locator template identifying the folder root. May be absolute (e.g. `"s3://bucket/prefix/"`) or relative to a higher-level context. |

<u>Notes</u>:

- The folder locator is treated as **opaque**—not assumed to correspond to a physical file system unless the backend specifies one.
- Compilers may normalize relative paths against a project, tenant, or entity context.
- Inheritance may only tighten or specialize `path` templates (e.g., narrow a prefix).

#### StateDescriptor(RelationDescriptor)  ← mixin: IdentifiableDescriptor (AML: `state`)

`StateDescriptor` declares a **time-varying value** for a model, backed by a record collection. A state is always **dynamic** (evaluated as of an event time) and always resolves to a `Data` value. By default, a state reads from an underlying time series and returns the value of that series **as of** a given time, optionally subject to lag/staleness and reduction rules. The underlying series must be *joinable* to the owning entity, typically via a `link` field from the record to the entity.

**Admissible owner:** `Entity`
**Admissible assigned types:** `DataType` (scalar or compound)

Typical usage patterns:

- From a record column:

  ```python
  class MotorTelemetry(nx.Record):
      motor: nx.link["Motor"]
      timestamp: nx.Datetime = nx.data(timestamp=True)
      temperature: float = nx.data(state=True)
  
  class Motor(nx.Entity):
      # Latest known temperature as of the query time
      temperature: float = nx.state(source=MotorTelemetry.temperature)
  ```

- From a record with a single state column:

  ```python
  class SensorReading(nx.Record):
      sensor: nx.link["Sensor"]
      timestamp: nx.Datetime = nx.data(timestamp=True)
      value: float = nx.data(state=True)
  
  class Sensor(nx.Entity):
      # `value` is the only state=True column; used implicitly
      value: float = nx.state(source=SensorReading)
  ```

- Implicit series (observation created by the framework):

  ```python
  class Sensor(nx.Entity):
      # Framework establishes an observation series for humidity
      humidity: float = nx.state(max_lag="10m")
  ```

- Custom reducer on a record series (cross-column, same-row):

  ```python
  class MotorTelemetry(nx.Record):
      motor: nx.link["Motor"]
      timestamp: nx.Datetime = nx.data(timestamp=True)
      temperature: float = nx.data()
      pressure: float = nx.data()
  
  class Motor(nx.Entity):
      health_index: float = nx.state(
          source=MotorTelemetry,
          reducer=lambda t, rows: (
              rows[-1].temperature * 0.7 + rows[-1].pressure * 0.3
          )
      )
  ```

- Custom reducer on a selection:

  ```python
  class Motor(nx.Entity):
      telemetry: nx.DataSequence[MotorTelemetry] = nx.selection(
          time=(nx.dt.lag("10m"), nx.dt.now)
      )
  
      avg_temperature_10m: float = nx.state(
          source=telemetry,
          reducer=lambda t, rows: (
              sum(r.temperature for r in rows) / len(rows)
          ),
          min_observations=3,
      )
  ```

Evaluation is always with respect to an event time, exposed at runtime via method arguments (e.g. `motor.temperature(time=...)`). If no time is supplied, the default is `nx.dt.now()`.

`StateDescriptor` accepts the following attributes, in addition to inherited ones:

| Argument           | Type                                                         | Default | Description                                                  |
| ------------------ | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
| `source`           | `type[Record] | DataDescriptor | SelectionDescriptor | None` | `None`  | Identifies the underlying time series. May be a record prototype, a record’s data descriptor (i.e. a column, filtered for a single key, which evaluates to a `SerialSequence`), a selection returning a `DataSequence`, or `None` to request that the framework establish an implicit observation series for this state. |
| `reducer`          | `str | Callable | None`                                      | `None`  | Optional reduction policy that transforms the time series into a state value. Pattern strings are only allowed when `source` uniquely identifies a value column. Callables are only allowed when `source` is a record prototype or a selection. |
| `max_lag`          | `str | timedelta | None`                                     | `None`  | Optional staleness threshold. If the most recent observation is older than this lag relative to the evaluation time, the state is treated as undefined. |
| `min_observations` | `int | None`                                                 | `None`  | Minimum number of observations required in the effective support set before a value is emitted. Primarily used with selection-backed reducers that operate on multiple rows. |

<u>Reducer resolution</u>:

- If `reducer` is **omitted**:
  - When `source` identifies a column (explicitly or via `state=True` in the source `Record` class), the default reducer is equivalent to `"asof"` (latest observation at or before the evaluation time).
  - When `source` is a record prototype without a unique state column, a default reducer is only applicable if the compiler can infer a value column; otherwise a reducer or explicit column must be supplied.
  - When `source` is a selection, a reducer is required if the selection may produce multiple rows.
- If `reducer` is a **string pattern**, it is only allowed when the value column is unambiguous:
  - `source` is a record data descriptor (e.g. `MotorTelemetry.temperature`), or
  - `source` is a record prototype with exactly one `DataDescriptor` flagged `state=True`, or
  - `source` is `None` (implicit observation with a single value column).
- If `reducer` is a **callable**, it is only allowed when:
  - `source` is a record prototype, or
  - `source` is a selection descriptor.

In all cases, the reducer is invoked with the evaluation time and an iterable of rows drawn from the underlying time series; the returned value must be compatible with the state’s declared `Data` type. If the effective support set is empty, or contains fewer than `min_observations` rows (when specified), the state is treated as undefined and the reducer is not called.

<u>Patterned reducers</u> (provisional list):

Pattern names are reserved for a small set of common behaviors on scalar or simple `Data` series, for example:

- `"asof"` — latest observation at or before the evaluation time (piecewise-constant state). This is the implicit, default behavior.
- `"nearest"` — observation whose timestamp is closest to the evaluation time, subject to `max_lag` if provided.
- `"linear"` — linear interpolation between neighboring observations around the evaluation time, where supported by the backend and type.
- `"avg"` — average over the support defined by the underlying selection or series (e.g. a trailing window).

Implementations may extend this list, but must preserve the semantics implied by these names.

<u>Notes</u>:

- States are always **dynamic**: evaluation depends on an event time, even when no explicit `time` argument is supplied by the caller.
- The underlying series must provide an event timestamp or equivalent ordering; compilers enforce the presence and compatibility of temporal metadata.
- The join between an entity and its state series is typically expressed via a `link` from the record model to the entity; more exotic join patterns should be modeled via `selection` descriptors that the state can then reference as `source`.
- `max_lag` and `min_observations` are enforced prior to reducer execution; if they are not satisfied, the state is considered undefined for that evaluation.
- Inheritance may narrow staleness and observation constraints (reduce `max_lag`, increase `min_observations`) but may not relax them.

#### FieldExpressionDescriptor(FieldDescriptor)  ← mixins: CallableDescriptor, IdentifiableDescriptor (AML: `fx`)

`FieldExpressionDescriptor` declares a **derived instance attribute** whose value is computed from other attributes via an expression. It is read-only and has no dedicated storage; compilers may materialize it as a computed column, view field, or in-memory property depending on backend capabilities.

Field expressions are defined with `nx.fx(...)` and accept a **single argument**, which is either:

- a **callable** (Python function/lambda) taking a single instance of the declaring model as its only argument, or
- a **string** used to reference another attribute (optionally with dotted notation).

**Admissible owner:** `Model`
**Admissible assigned types:** `ObjectType`

Typical usage patterns:

- Callable form on the same model:

  ```python
  class Order(nx.Entity):
      ...
      subtotal: float = nx.data()
      tax: float = nx.data()
      total: float = nx.fx(lambda o: o.subtotal + o.tax)
  ```

- Callable form using related models:

  ```python
  class OrderItem(nx.Record):
      order: "Order" = nx.link(key=True)
      quantity: int = nx.data()
      unit_price: float = nx.data()
  
  class Order(nx.Entity):
      ...
      items: nx.DataMapping[OrderItem] = nx.selection()
      line_count: int = nx.fx(lambda o: len(o.items))
  ```

- String form referencing a local attribute:

  ```python
  class Sensor(nx.Entity):
      ...
      value: float = nx.fx("raw_value")
      raw_value: float = nx.data()
  ```

- String form with dotted notation:

  ```python
  class Device(nx.Entity):
      ...
      site: "Site" = nx.link()
      location_name: str = nx.fx("site.name")
  ```

- Decorator form:

  ```python
  class Order(nx.Entity):
      ...
      subtotal: float = nx.data()
      tax: float = nx.data()
  
      @nx.fx
      def total(self) -> float:
          return self.subtotal + self.tax
  ```

In all cases, evaluation occurs on the owning instance; when a callable is supplied, its first and only parameter is the instance whose field is being computed.

`FieldExpressionDescriptor` accepts the following attributes, in addition to inherited ones:

| Argument | Type             | Default    | Description                                                  |
| -------- | ---------------- | ---------- | ------------------------------------------------------------ |
| `expr`   | `Callable | str` | *required* | Expression used to derive the field value. A callable is invoked with the owning instance as its sole argument. A string is interpreted as an attribute reference, optionally using dotted notation to traverse links or other attributes. |

<u>Expression semantics</u>:

- The field’s type is inferred from the callable’s return annotation (callable form) or from the referenced attribute’s type (string form). An explicit type hint on the field is required when inference is ambiguous.
- Field expressions are **read-only**; assignment to them is invalid.
- Expressions must be side-effect free and deterministic with respect to their inputs. They may only depend on attributes and relations accessible from the owning instance.
- Cyclic dependencies between field expressions (direct or indirect) are undefined and rejected by compilers.

<u>Stateful vs non-stateful expressions</u>:

- A field expression is **non-stateful** if it only depends on:
  - data fields (`nx.data`),
  - links/backlinks (`nx.link`, `nx.backlink`),
  - other non-stateful field expressions.
- A field expression is **stateful** if it references:
  - a `StateDescriptor`,
  - a dynamic `SelectionDescriptor` (i.e. one whose value depends on an event time),
  - or another stateful field expression.

Non-stateful field expressions are invariant with respect to event time and may be materialized as static computed columns where supported. Stateful field expressions are dynamic and inherit the event-time semantics and evaluation context of the stateful attributes they depend on.

<u>Lowerable vs runtime-only expressions</u>:

Compilers may attempt to **lower** field expressions to backend-native constructs (e.g. SQL expressions, computed columns, or view fields). To be considered for lowering, an expression must obey additional constraints, such as:

- Use of attribute access, basic arithmetic, comparisons, and boolean combinations that can be mapped to the target backend.
- Use of aggregations or functions that are either intrinsic to the backend or explicitly supported by the compiler.
- No use of arbitrary Python control flow or side-effects that cannot be translated.

Expressions that fall outside the lowerable subset are still valid, but are evaluated in memory after data has been loaded.

<u>Use in other descriptors</u>:

`FieldExpressionDescriptor` (or inline `nx.fx(...)` expressions) may be used as **inputs** to other descriptors and schema constructs that accept expressions, for example as a filter or sort expression in selections:

```python
class Order(nx.Entity):
    ...
    open_items: nx.DataSequence[OrderItem] = nx.selection(filter=nx.fx(lambda o: o.status == "open"))
```

When used on the right-hand side of a field definition, `nx.fx(...)` declares a named `FieldExpressionDescriptor`. When used inline as an argument to another descriptor (e.g. `selection(filter=nx.fx("..."))`), it is treated as an expression object only and is not registered as a separate field.

**<u>Provisional lowerable subset for field expressions:</u>**

The exact lowerable subset is backend-dependent. Backends may support additional functions or constructs beyond this provisional list, but should not support less.

- **Attribute access**
  - Access to fields and relations on the owning instance:
    - `obj.field`
    - `obj.link.field`
    - `obj.selection` (where the selection’s value is supported by the target backend)
  - No arbitrary attribute access outside the model graph.
- **Literals**
  - `int`, `float`, `bool`, `str`, `None`
  - Backend-supported date/time and timedelta literals where applicable.
- **Numeric operations**
  - `+`, `-`, `*`, `/`, `//`, `%`, unary `-`
  - On numeric `Data` types and backend-supported numeric types.
  - Optional: `+` / `-` between timestamps and timedeltas, if supported by the backend.
- **Comparisons**
  - `<`, `<=`, `>`, `>=`, `==`, `!=`
  - Chained comparisons allowed if the backend supports them, otherwise decomposed.
- **Boolean logic**
  - `and`, `or`, `not`
  - Combinations of boolean-valued expressions (including comparisons).
- **Membership tests (optional)**
  - `x in (a, b, c)` / `x in [a, b, c]` as `IN` where supported.
- **Simple built-ins and functions**
  - `abs(x)`, `round(x)`, `min(x, y, ...)`, `max(x, y, ...)`, `len(x)` on supported collections.
  - A small set of math/date functions if you want, e.g. `nx.date_trunc`, `nx.coalesce`, `nx.nullif`, provided they have a direct backend mapping.
- **No side effects / no control flow in the lowered subset**
  - No `for`, `while`, `if`/`else` statements in callable bodies if you require pure expression form for lowering (or: `if` only when it maps to `CASE` expressions).
  - No mutation, I/O, or global state.

Anything outside this subset is still legal as an `fx`, but is classified as **runtime-only** and evaluated in Python after materialization.

#### FieldGroupDescriptor(FieldDescriptor)  ← mixins: IdentifiableDescriptor, FieldListDescriptor (AML: `fieldgroup`)

`FieldGroupDescriptor` declares a **named group of fields** on a model. Field groups are used for reusable projections, query presets, and higher-level schema annotations (e.g. “identity fields”, “summary fields”). They do not introduce new storage; they simply reference existing fields.

Field groups may contain any kind of field descriptor, including stateful attributes (e.g. states), and may also be defined in terms of other field groups.

**Admissible owner:** `Model`
**Admissible members:** `FieldDescriptor` and `FieldGroupDescriptor`

Typical usage patterns:

```python
class Motor(nx.Entity):
    name: str = nx.data()
    serial_number: str = nx.data()
    installed_at: nx.Datetime = nx.data()
    location: str = nx.data()

    identity = nx.fieldgroup(name, serial_number)
    installation = nx.fieldgroup(installed_at, location)
    # Multiple groups can reference the same field
    summary = nx.fieldgroup(name, serial_number, location)
    # And groups can be defined with other groups (flattened)
    # This formulation is equivalent to the one above
    summary = nx.fieldgroup(identity, location)
    ...
    status: str = nx.state(...)
    health_index: float = nx.state(...)
	# Group containing stateful fields
    status_panel = nx.fieldgroup(status, health_index)
```

In all cases, `nx.fieldgroup(...)` is called at class definition time with one or more members, each being either a field descriptor or another field group.

<u>Notes</u>:

- A field group defines a **logical projection**. It does not affect storage layout directly, but may be used by compilers and APIs as a convenient way to select or materialize a subset of fields.
- Members of a group may be:
  - individual fields (data, link, state, field expressions, etc.), or
  - other field groups. When a group is used as a member, its `fields` are **flattened** into the new group in declaration order.
- When flattening, duplicates are removed while preserving the first occurrence. This ensures deterministic ordering even when groups overlap.
- Cyclic group definitions (direct or indirect) are invalid and rejected at model-finalization time.
- A field may belong to multiple groups; membership is determined solely by the `nx.fieldgroup(...)` declarations on the model.
- Inheritance may refine groups by:
  - adding or removing fields,
  - changing `doc`, while preserving the general tightening semantics of the model.

#### FieldBlockDescriptor(FieldDescriptor)  ← mixin: AnnotatableDescriptor (AML: `fieldblock`)

`FieldBlockDescriptor` declares that the fields of another model are **inserted into** the declaring model. It is a structural shortcut: the target model’s fields are exposed directly on the owner, and a field group named after the block is created to reference those fields. For instance:

```python
class Foo(nx.Model):
    a: int = nx.data()
    b: int = nx.data()

class Bar(nx.Model):
    foo: Foo = nx.fieldblock()
    # After compilation, Bar has:
    #   a: int
    #   b: int
    # and a field group named "foo" containing [a, b]
```

**Admissible owner:** `Model`
**Admissible assigned types:** `ModelType`

<u>Notes</u>:

- Only eligible fields from the source model are inlined (typically data, link, state, and field-expression fields; schema descriptors and other blocks are not inlined as fields).
- The block’s internal fields are expanded **in place**, at the position where the block is declared.
- Inlining copies the **logical field declarations** (names, types, and relevant flags) into the owner model. Backends may choose to share or reuse physical storage where appropriate.
- For each field block `X` declared on a model:
  - fields from the source model are inlined into the owner, subject to the `fields` filter, and
  - a `FieldGroupDescriptor` named `X` is defined, containing the inlined fields in their declaration order.
- Name collisions between inlined fields and existing fields on the owner are invalid and rejected at model-finalization time unless explicitly resolved by the compiler or configuration.
- Field blocks can be used repeatedly across models to enforce consistent sets of fields (e.g. audit metadata, location descriptors, common measurement layouts).
- Inheritance may refine blocks by:
  - narrowing the `fields` list,
  - adjusting `doc`, while preserving the basic inlining semantics.
- Block expansion happens during **resolution**, after declarator merging.
- Field blocks **do not affect precedence rules**; they only affect ordering.
- The only problematic cases are **name collisions**.

<u>Validity Rules</u>:

1. A field block may only introduce **new field names**.
2. If a block introduces a field name that already exists **at or before the expansion point**, raise.
3. Multiple blocks must not introduce overlapping field names; raise.
4. Fields introduced by a block **may be overridden later** (override preserves position).
5. A block may be inherited; expansion happens once and is inherited structurally.
6. A block may be overridden by another block; the new block expands at the same position.


<u>Resolution Algorithm (Field Ordering + Block Expansion)</u>:

- Merged field declarators are processed in declaration order  
  (`archetype → traits → declaring class`).
- Each declarator is either a `field` or a `fieldblock`.
- Each `fieldblock` references a block model with its own resolved field order.

```python
resolved_fields = []
seen = set()

for decl in merged_declarations:
    if decl.is_field:
        name = decl.name
        if name in seen:
            continue  # override, keep position
        resolved_fields.append(name)
        seen.add(name)

    elif decl.is_fieldblock:
        block_fields = decl.block_model.resolved_field_order
        for name in block_fields:
            if name in seen:
                raise ValueError(
                    f"Field block introduces duplicate field '{name}'"
                )
            resolved_fields.append(name)
            seen.add(name)
```

#### MetricDescriptor(FieldDescriptor)  ← mixins: CallableDescriptor, IdentifiableDescriptor (AML: `metric`)

`MetricDescriptor` declares a **derived value on an aggregate**, computed from the elements of that aggregate. Metrics are read-only, have no dedicated storage, and are typically defined on prototypes whose archetype is an aggregate (e.g. `DataSequence[T]`, `DataMapping[K, T]`). Compilers may materialize metrics as aggregate expressions, grouped queries, or in-memory reductions depending on backend capabilities.

Metrics are defined using `nx.metric(...)` or the `@nx.metric` decorator and accept a single expression argument, a callable taking the aggregate instance as its sole argument.

**Admissible owner:** `AggregateType`
**Admissible assigned types:** `DataType` (scalar or compound, non-aggregate)

Typical usage patterns:

- Decorator form:

  ```python
  class VehicleLog(nx.DataSequence["VehicleRecord"]):
      @nx.metric
      def max_speed(self) -> float:
          return max(rec.speed for rec in self)
  ```

- In-line form:

  ```python
  class OrderItems(nx.DataSequence["OrderItem"]):
      total_value: float = nx.metric(lambda seq: sum(i.quantity * i.unit_price for i in seq))
  ```

<u>Empty-aggregate semantics:</u>

Before invoking the metric expression, the framework applies a uniform rule:

- If the aggregate is **empty** and the metric’s declared type allows `None`, the result is `None`.
- If the type is **non-nullable**, an error is raised (e.g. `MetricEmptyError`).
- The metric expression is **not invoked** for empty aggregates.

This guarantees that metric implementations do not need to defend against empty inputs.

<u>Expression semantics:</u>

- The metric’s type is inferred from the callable’s return annotation.
- Expressions must be side-effect free and deterministic with respect to their inputs.
- Cyclic metric dependencies (direct or indirect) are invalid and rejected at model-finalization time.

<u>Lowerable vs. runtime-only metrics:</u>

Compilers may lower metric expressions to native backend operations (e.g., `SUM`, `AVG`, `COUNT`, `MAX`, `MIN`) when the expression falls within a recognized subset, such as:

- reductions over element attributes (`sum(...)`, `min(...)`, `max(...)`),
- length (`len(self)`),
- arithmetic compositions (e.g. `sum(...) / len(self)`),
- simple boolean comparisons or combinations of aggregated values.

Expressions outside this subset remain valid but are evaluated in memory once the aggregate has been materialized.

<u>Usage with selections:</u>

Metrics defined on aggregate prototypes are available wherever those aggregates appear:

```python
class OrderItem(nx.Record):
    order: nx.link["Order"]
    quantity: int = nx.data()
    unit_price: float = nx.data()

class OrderItems(nx.DataSequence[OrderItem]):
    @nx.metric
    def total_value(self) -> float:
        return sum(i.quantity * i.unit_price for i in self)

class Order(nx.Entity):
    items: nx.DataSequence[OrderItem] = nx.selection(key="order")
    # items.total_value is now available
```

Backends may expose metrics as projected fields, computed columns, or query-level aggregates, while preserving the semantics defined here.

### MethodDescriptor (abstract) ← mixins: CallableDescriptor

`MethodDescriptor` is the abstract base class for descriptors that wrap methods declared on prototypes. These descriptors introduce callable logic into model definitions using decorators and are primarily used for input transformation and data validation. Method descriptors remain declarative: they are collected by the prototype metaclass and forwarded to compilers, which integrate them into the generated interfaces and validation schemas.

Method descriptors may appear on any prototype unless otherwise noted. Their callable component is always stored for later compilation and is never executed during prototype creation.

A `MethodDescriptor` wraps a Python callable supplied either by using the descriptor as a decorator on a method or by providing a callable inline through a field definition. The descriptor attaches to the method name under which it is declared. Method descriptors do not participate in instance layout and introduce no storage fields.

#### ConstructionDescriptor(MethodDescriptor) (abstract)

Construction descriptors define functions invoked during parsing or validation of structured data. They may apply to individual fields or to entire objects, depending on their declaration form.

Parsers and validators participate in a consistent construction pipeline that respects the ownership hierarchy of the object graph. For a given owner instance and a given owned value (for example, a nested model or aggregate), the effective order is:

1. Owner-level parsers for the field being populated
2. Owned-type parsers declared on the prototype of the nested value
3. Owned-type validators declared on the prototype of the nested value
4. Owner-level validators (field-level and whole-object) declared on the owner prototype

Within each tier, multiple parsers or validators are applied in deterministic declaration order.

This composition rule applies recursively along nesting: each level in the object graph observes the same precedence between its own construction descriptors and those of the types it owns.

#### ParserDescriptor(ConstructionDescriptor) (AML: `parser`)

A `ParserDescriptor` declares a transformation applied to raw inputs. Parsers normalize, coerce, or otherwise convert values before validation and materialization. Parsers may target:

- a specific field or set of fields, or
- the whole object.

**Declaration forms**

1. Decorator form

   ```python
   class Order(nx.Entity):
       amount: float = nx.data()
   
       @nx.parser("amount")
       def normalize_amount(cls, value: float) -> float:
           ...
   ```

2. Field keyword form

   ```python
   class Order(nx.Entity):
       amount: float = nx.data(parser=lambda value: round(value, 2))
   ```

**Attributes**

- `fields: tuple[str, ...] | None` — names of targeted fields, or `None` for whole-object parsers
- `callable: Callable` — the parser function
- `element_wise: bool = False` — when the owner archetype is a `DataFrame` or subtype, indicates whether the parser applies element-wise

**Callable signatures**

- **Field parser (decorator):**
   `def parse_<name>(cls, value, /, ...) -> Any`
   `cls` is the owning prototype; `value` is the field’s incoming value.
   When multiple field names are listed, the callable is applied to each field independently, always with a single `value` argument.
- **Field parser (keyword):**
   `Callable[[Any], Any]`
- **Whole-object parser:**
   a callable receiving the object-level input (typically a mapping or intermediate representation)

Parsers return the transformed value for the targeted field or the transformed object for whole-object parsers.

**Semantics**

- Parsers run before validators
- Parsers must return the transformed value (or object)
- Parsers are applied in deterministic declaration order

#### ValidatorDescriptor(ConstructionDescriptor) (AML: `validator`)

A `ValidatorDescriptor` declares a Boolean predicate used to validate field values or fully constructed objects. Validators may target:

- one or more fields, or
- the whole object.

**Declaration forms**

1. Decorator form, field-level

   ```python
   class Order(nx.Entity):
       amount: float = nx.data()
   
       @nx.validator("amount")
       def positive_amount(cls, value: float) -> bool:
           return value > 0
   ```

2. Decorator form, object-level

   ```python
   class Order(nx.Entity):
       @nx.validator
       def totals_ok(self) -> bool:
           ...
   ```

3. Field keyword form

   ```python
   class Order(nx.Entity):
       amount: float = nx.data(validator=lambda value: value > 0)
   ```

**Attributes**

- `fields: tuple[str, ...] | None` — names of targeted fields, or `None` for whole-object validators
- `callable: Callable` — the validation function
- `element_wise: bool = False` — when the owner archetype is a `DataFrame` or subtype, indicates whether validation is applied element-wise

**Callable signatures**

- **Field validator (decorator):**
   `def validate_<name>(cls, value, /, ...) -> bool`
   `cls` is the owning prototype; `value` is the field’s value.
   When multiple field names are listed, the callable is applied to each field independently, always with a single `value` argument.
- **Field validator (keyword):**
   `Callable[[Any], bool]`
- **Whole-object validator:**
   `def validate(self, /, ...) -> bool`
   `self` is the model instance.

Validators return `True` when the constraint holds and `False` otherwise.

**Semantics**

- Validators operate on the values produced by associated parsers
- Validators may be composed; compilers aggregate all validators declared on the prototype
- Validators do not modify data

### SchemaDescriptor

`SchemaDescriptor` defines schema-level elements that govern identity, ordering, uniqueness, indexing, partitioning, and logical addressing. They are practically mostly relevant to `Model` archetypes, with the exception of the path descriptor, which has broader application. Schema descriptors are declared at the prototype level and are consumed by compilers (ORM, columnar, graph, file-like, etc.) to generate physical schemas and access paths. They are declared anonymously, inside the special `__nxschema__` list. This attribute is a class-level list of schema descriptor instances constructed by calling their AML functions:

```python
class VibrationSample(nx.Record, nx.sample):
    ...
    __nxschema__ = [
        nx.unique(probe, timestamp),
        nx.path(t"{probe}/{timestamp}", name="probe"),
    ]
```

- **Scope:** prototype-level
- **Admissible owners:** `Object`-derived structural types (typically `Model`, `Entity`, `Record`, `Document`)
- **Common attribute pattern:** most schema descriptors refer to one or more fields and implement the `FieldListDescriptor` mixin.

As a general rule:

- Field-level flags (`key=True`, `sequence=True`, `unique=True`, `index=True`, etc.) on `DataDescriptor` and `LinkDescriptor` are **syntactic shortcuts** for schema descriptors.
- Explicit schema descriptors are **more precise** and prevail in ambiguous or conflicting cases; detectable conflicts are reported as errors.
- `__nxschema__` is **optional**. When present, it must be an iterable of schema descriptor instances created by calling the corresponding AML functions (`nx.key`, `nx.unique`, `nx.index`, `nx.partition`, `nx.path`, `nx.sort`, etc.).
- Schema descriptors are treated exactly like named descriptors for the purpose of compilation, but they do not occupy a named attribute in the prototype’s namespace.
- Inheritance applies as usual: a child prototype inherits all schema descriptors from its parents and may add additional descriptors.

#### KeyDescriptor(SchemaDescriptor) ← mixin: FieldListDescriptor (AML: `key`)

`KeyDescriptor` declares one or more fields as a primary identity key for a prototype. Entities and records may each define a single key descriptor, but they need not to. In the case of entities, the primary key serves as a user-facing unique key, and compiles to a unicity constraint and index, but the framework automatically adds a UUID to serve as the physical primary key either way. Records are uniquely identified by a combination of key and sequence, both of which are optional -so the declared key may or may not end up as a physical primary key.

- **Admissible owner:**  `Entity`, `Record`
- **Admissible members:** `DataDescriptor`, `LinkDescriptor` 

**Notes**

- Field-level `key=True` on `nx.data` / `nx.link` is equivalent to declaring a `KeyDescriptor` that includes these fields. Mixed usage is allowed, but any contradictions raise an error.
- Key fields are always non-nullable; including a nullable field in a key is invalid.

#### SequenceDescriptor(SchemaDescriptor) ← mixin: FieldListDescriptor (AML: `sequence`)

`SequenceDescriptor` defines the field or fields that establish the **ordering dimension** for immutable records, most commonly an event timestamp.

- **Admissible owner:** `Record`
- **Admissible members:** `DataDescriptor`, `LinkDescriptor`

**Notes**

- Field-level `sequence=True` on a data field is equivalent to declaring a `SequenceDescriptor` containing that field.
- Sequence fields are implicitly indexed and must be non-nullable.
- Temporal traits (`sample`, `event`, `journal`, `session`, `phase`) may require or constrain sequence descriptors; compilers enforce the presence and compatibility of sequence information.

#### UnicityDescriptor(SchemaDescriptor) ← mixin: FieldListDescriptor (AML: `unique`)

`UnicityDescriptor` declares a **uniqueness constraint** on a combination of fields. It may be used for alternate keys, partial functional dependencies, or to mirror backend-level unique constraints.

- **Admissible owner:** `Model`
- **Admissible members:** `DataDescriptor`, `LinkDescriptor`

**Notes**

- Field-level `unique=True` on a descriptor emits a `UnicityDescriptor` for that field, subject to nullability rules.
- For nullable fields, uniqueness is interpreted in the usual database sense: duplicate combinations are disallowed **across non-null values**; combinations containing `NULL` are exempt unless a particular backend offers stricter semantics.
- Multiple `unique` descriptors may coexist on the same model, each defining a distinct unique constraint.
- Inheritance may add new uniqueness constraints but must not remove or weaken existing ones.

#### IndexDescriptor(SchemaDescriptor) ← mixin: FieldListDescriptor (AML: `index`)

`IndexDescriptor` declares a **secondary index** used to accelerate lookups or range queries on one or more fields. Indexes are optional hints; compilers interpret them according to backend capabilities.

- **Admissible owner:** `Model`
- **Admissible members:** `DataDescriptor`, `LinkDescriptor`

**Attributes**

- `kind: str | None = None` — Optional index kind to specify a physical indexing strategy.

**Notes**

- Field-level `index=True` is equivalent to an `IndexDescriptor` on that field with default `kind`.
- Compilers map `kind` to backend-specific constructs (e.g. B-tree, hash, GIN, GiST). Unsupported kinds either fall back to a default or raise a configuration error.
- Composite indexes are created when `fields` lists multiple names.
- Inheritance may add new indexes or refine `kind`, but must not remove indexes required by other constraints (e.g. for enforcing uniqueness).

#### PartitioningDescriptor  (AML: `partition`)

`PartitioningDescriptor` declares a partitioning strategy over a **single partitioning key** used to segment stored models for physical or logical layout. The key may be a stored field or a scalar field expression. The partitioning scheme determines the admissible type of the key and how buckets are interpreted. Compilers translate partitioning descriptors into backend-specific mechanisms when supported.

- **Admissible owner:** `Model`

**Attributes**

- `key: DataDescriptor | LinkDescriptor | FieldExpression` — The expression’s result type must satisfy the scheme-specific constraints (ordered, temporal, finite-domain, or hashable).
- `scheme: Literal["range", "categorical", "hash", "time"]`
- `buckets: None | int | tuple | list | str` — Interpreted according to `scheme` and validated for consistency.

**Partitioning schemes**

**`scheme="range"`**
 Range partitioning over an ordered domain.

- `buckets`:
  - list of strictly increasing edges (`[e1, e2, …, ek]`), or
  - tuple `(start, stop, step)` (half-open; `stop` exclusive).
- Semantics: `[edge_i, edge_{i+1})` with a final bin `[edge_k, +∞)`; tuple form produces `[start + n*step, start + (n+1)*step)`.
- Key type must be ordered (numeric or temporal).

**`scheme="time"`**
 Calendar-aligned temporal partitioning.

- `buckets`: `"year" | "quarter" | "month" | "week" | "day" | "hour"`.
- Semantics: partitions correspond to aligned calendar intervals.
- Key type must be temporal (`date`, `datetime`, or timestamp).

**`scheme="categorical"`**
 Partitioning over a finite, ordered domain.

- `buckets`:
  - `None` → one partition per category, or
  - integer `K` → contiguous grouping into `K` bins.
- Domain ordering: enums follow declaration order; lookups/strings use lexicographic order.
- Key type must be finite-domain.

**`scheme="hash"`**
 Hash-based partitioning using a stable, deterministic hash.

- `buckets`:
  - integer `K ≥ 1`, or
  - `None` only if the key is categorical (interpreted as `K = number_of_categories`).
- Semantics: `partition_id = stable_hash(key) % K`.
- Key type must be hashable.

**Validation and cross-cutting rules**

- Null key values are not assigned to a data partition; compilers may use a backend-specific null partition if supported.
- Bucket specifications must match the scheme: monotone edges; valid `(start, stop, step)` tuple; integer counts `≥ 1`.
- Categorical schemes require a verifiably finite domain.
- Partitioning is singleton per hierarchy; differing semantics require a distinct descriptor name.

#### **PathDescriptor  (AML: `path`)**

`PathDescriptor` declares a **unique identity path** for instances of a model. Identity paths are used by the DTI to address and reference individual instances across APIs, artifact stores, and document layers. Path templates are **Python template strings** (type `str`) interpreted by AML according to its path-template grammar, using `{field}` and `{fk.chain}` placeholders to build hierarchical, human-readable identifiers.

- **Admissible owner:** any model whose instances have independent DTI identity. This includes `Entity`, `Record`, and `Document` models, as well as any prototype implementing the `artifact` trait (including `Media`). Inline or embedded models must not declare paths.

**Attributes**

- `template: str` — A Python template string containing an AML path template. Placeholders of the form `{field}` or `{fk.chain}` must resolve to declared fields or linked models; parent identity paths may be incorporated via foreign-key expansion.
- `name: str | None = None` —  Optional symbolic name. If a model defines exactly one unnamed path, it overrides the canonical identity path. If `name="canonical"`, the override is explicit. Other names define alternate identity routes (e.g. station-based vs. sample-based addressing).

**Uniqueness**

A `PathDescriptor` must define a **unique identity path** for each instance of the model on which it is declared. In the motor example:

```python
nx.path(t"{nickname}", name="nickname")
```

is valid because `nickname` is declared `unique=True`, ensuring that `{nickname}` is injective over all motor instances.

Likewise, for vibration samples:

```python
nx.path(t"{probe}/{timestamp}", name="probe")
```

is valid because `(probe, timestamp)` is constrained to be unique (across non-null probe values). Any template that can expand two distinct instances to the same path must be rejected.

**Foreign-key chain expansion**

Templates may reference linked models using placeholders such as `{monitoring_location}` or `{probe}` in the motor example. If the referenced model defines its own identity path, that identity path is expanded in place. For example, a sample path:

```python
t"{probe}/{timestamp}"
```

expands with the probe’s identity (derived from its model and serial attributes) followed by an ISO-formatted timestamp, producing paths such as:

```python
"Acme/17/2025-04-02T07-30-00"
```

Uniqueness composes transitively: a unique parent identity path combined with a suffix that uniquely identifies the child yields a unique child identity path.

**Canonical vs. named paths**

Every model has an implicit canonical identity path derived from its key and storage strategy. A `PathDescriptor` may redefine it:

- One unnamed path ⇒ replaces the canonical path.
- `name="canonical"` ⇒ explicitly replaces the canonical path.
- Additional named paths coexist and define alternate identity routes.
- All declared paths must satisfy the uniqueness rule for the model.

**Template-string rationale**

AML uses template strings (`t"..."`) instead of raw strings or Python expressions because they provide:

- **Static analyzability**: placeholders are validated, FK chains traversed, uniqueness checked.
- **Hierarchical composition**: parent identity paths embed cleanly via `{fk}`.
- **Purity and portability**: templates are declarative schema objects, serializable and backend-agnostic, not Python code.

**Validation rules**

- All placeholders must reference declared fields or linked models.
- `{fk}` implies that the referenced model must itself define a unique identity path.
- The resulting expanded template must be injective over the model’s key.
- Formatting must respect declared field types (e.g. timestamps).
- Path descriptors are invalid on models without independent DTI identity.

#### SortDescriptor(SchemaDescriptor)  (AML: `sort`)

`SortDescriptor` declares a **default ordering** for instances of a model. It is used as a stable, conventional ordering in query results, user interfaces, and materialized views when no explicit sort is requested.

- **Admissible owner:** `Model`

**Attributes**

- `sortkeys: str | Sequence[str]` — One or more sort keys. Each key may be prefixed with `"-"` to indicate descending order (e.g. `"timestamp"` or `["-timestamp", "id"]`).

**Notes**

- If no `SortDescriptor` is defined, a backend or compiler-specific default ordering may apply (commonly primary key).
- Sort keys should be index-compatible where possible (e.g. overlapping with key, sequence, or indexed fields) to avoid inefficient ordering.
- Inheritance may refine sorting by:
  - reusing the parent’s sort order,
  - prepending or appending additional sort keys,
  - or tightening sort directions (e.g. replacing an implicit ascending with an explicit descending when compatible with usage).

## Implementation notes

### Language Implementation

#### Core Package

The core package contains the fundamental building blocks of the Anaximander Modeling Language (AML). It is designed to be functionally independent from the definition of the language itself. Instead, the core package provides infrastructure aimed at writing a domain-specific language that makes the following assumptions:

- The language is declarative in nature. Its declaration units are Python classes called declaratives. Declaratives cannot be instantiated directly, but they serve as the source for runtime classes that are created through either compilation or metaprogramming. Concretely, `declarative` is a metaclass that collects declarator statements from class bodies, validates them, and organizes them into structured collections for further processing. Their mechanics are facilitated by a set of declarator objects, inheriting from the `Declarator` base class, and a `Registry` class that serves to register, bind, and query declarators.

- One particular type of declarative is the `prototype` (an inherited metaclass). Prototypes are the core building blocks of the language, in that they are used to define the application domain. In AML, this in contrast with `nxtype` (another metaclass that inherits from `declarative`), which is used to specify system behavior using a combination of declarators and straight Python code. 

- Prototypes are organized in a type hierarchy. That hierarchy is comprised of archetypes, traits, and genuine prototypes. Archetypes define the structure and behavior of the prototypes that inherit from them, while traits provide additional modular functionality that can be mixed into prototypes. Genuine prototypes are the concrete models that can be compiled or transformed into runtime classes. A prototype must inherit from exactly one archetype as its first base, either directly or through inheritance from a parent prototype, along with zero or more traits. The traits themselves must be compatible with the base archetype, i.e., they must inherit from the same archetype or one of its bases. Archetypes can also define metacharacters, which then become class keyword arguments for all prototypes that inherit from them (i.e., similar to setting kwargs on `__init_subclass__`).

- Prototypes, archetypes, and traits are all defined as Python classes, and share a common metaclass called `prototype`. The body of prototype, archetype or trait classes is made up of declarative statements — in effect attributes, properties and methods — using declarators. Declarator registries associated with archetype determine how declarators are declared in a class body, how they are validated, whether and how they can be overridden in derived prototypes, whether and how they can be bound to a set value, etc. Declarators are organized in a class hierarchy that reflects their semantic meaning in the language definition, which in turn conditions how they are interpreted and compiled into runtime and system code. The declarators are split into two categories: protodescriptors and metadescriptors. Protodescriptors are used to declare attributes and methods in prototypes — in particular, protodescriptors belong to the semantic domain of the application that is being modeled with prototypes. Metadescriptors span all other aspects, including class-level metadata described in archetypes and traits, system attributes (for instance schema indexes or runtime options), and any other non-domain-specific construct.

- The body of prototypes consists of protodescriptor statements. Protodescriptors declare attributes much like Python descriptors, but they do not implement the descriptor protocol since prototypes are not instantiated directly. Instead, protodescriptors serve to collect metadata about the attributes they declare, which gets compiled into attribute definitions in system code libraries.

- Archetypes and traits are declared using decorators. These decorators relax inheritance constraints in order to allow injection of built-in or library classes in the inheritance hierarchy. The purpose of doing so is to instruct type checkers to treat derived prototypes as instances of those built-in or library classes, which greatly facilitates domain model development by providing the benefits of static type checking and IDE auto-completion. The core package provides the `archetype` and `trait` decorators for that purpose. The body of archetypes and traits is populated with metadescriptors. Like protodescriptors, these are declarator classes that do not implement the descriptor protocol, but instead collect metadata about the attributes they declare. Archetypes can create declarator registries, which de facto bind the type of declarators that can be declared in their derived prototypes. Both archetypes and traits can define prototype validation methods for new prototypes that inherit from them, using the `@prototype.validator` decorator.

- The core package also provides compilation infrastructure that allows prototypes to be transformed into runtime classes. This includes collection of declarators by the `prototype` metaclass, as well as filtered and structured AST representations of prototype bodies. However, the actual compilation process is left to the specific language implementation, since it is highly domain-specific.

- Finally, the core package provides basic error handling and reporting mechanisms to facilitate debugging and development of domain models.

#### The Declarator Class

The `Declarator` class is the abstract base class for protodescriptors and metadescriptors. It defines a set of interface options and behaviors for making declarations in class bodies. This includes:
- setting name, owner, assignability, annotated type. Specifically, Declarator is subclassed to reflect different semantics: assignable attributes, properties, typed attributes, callables, etc.
- Declarator has a `__validate__` method that can be called by `declarative` to perform validation on the declarator itself, for instance to check that its configuration is consistent. This method is called once when the declarator has been collected from a class body and its attributes finalized.
- adding validation methods for binding values to a declarator. This is realized by overriding an `__on_binding__` method, which defaults to raising an error. The `__on_binding__` method also accounts for the case when a declarator has already been bound in a parent class.
- specifying override rules, i.e., whether and how a declarator may be overridden in derived classes. This is realized by overriding an `__on_override__` method, which defaults to raising an error.
- There is also an `__on_compilation__` method that is executed after the module in which the Declarator's host class has been imported.

#### The Registry Class ####

The `Registry` class is designed to be instantiated by a `declarative` class instance, and applies to all that class' descendants. In particular, the `Arche` base class  declares the `__protodescriptors__` and `__metadescriptors__` registries. Registries store declarators and expose registration, binding and query interfaces. A registry is created by specifying a declarator source and admissible declarator types. For instance, the protodescriptors registry expects Protodesriptor instances, and looks exclusively in the class body. Registries can be attached to a parent registry to form namespace hierarchies. For instance in AML, the `Model` archetype adds a registry for schema descriptors, and makes it a subregistry of its metadescriptors registry. 

#### The `declarative` Metaclass

The `declarative` metaclass provides the following facilities:
- collects declarators from class headers and bodies — including collection attributes and inner classes, as well as assignments binding values to declarators;
- enforces assignability, validation, and override rules on declarators, including rules that are specified by archetypes and traits, starting with which declarators can appear in which context;
- stores the declarators and their bindings in registries, and provides a query interface for filtering by type, namespace, and a flag for merging declarations across the inheritance chain;
- `declarative` is designed with a customized `__prepare__` method that validates the namespace's declarations before they are further processed. It then calls declarator methods, including `__validate__`, and `__on_binding__` / `__on_override__` as appropriate.
- `declarative` also runs class-level validation checks -in the case of `prototypes`, these are defined and accumulated with the `@prototype.validator` decorator. 

### Executable References and Import Safety

#### Scope

This section defines the constraints on **executable code** that may appear in Anaximander Modeling Language (AML) modules, and the conditions under which AML modules are considered **import-safe**.

These constraints exist to ensure that:

- AML modules are evaluated **only at compile time**
- Generated artifacts are **self-sufficient at runtime**
- Compilation is **deterministic and reproducible**
- Regeneration and patching workflows remain tractable

#### Import Safety

AML modules are **executed by the compiler at import time** in order to declare prototypes, archetypes, traits, and associated descriptors.

Therefore, AML modules **MUST** satisfy the following import-safety requirements:

1. **No external side effects**  
   Importing an AML module MUST NOT:
   - perform I/O (filesystem, network, environment access)
   - read configuration or environment variables
   - depend on system time, randomness, or external state
   - mutate global state outside NX-controlled registries

2. **Deterministic declarations**  
   Given the same AML source and compiler version, importing the module MUST produce the same declarations every time.

3. **No runtime dependency on AML modules**  
   AML modules are compile-time artifacts only. Runtime systems MUST NOT import AML modules to obtain executable objects.

Violations of these rules render the AML module invalid.

#### Executable References in Declarations

AML declarations may reference executable objects (e.g. callables used as defaults, validators, reducers, selectors).  
Such references are permitted **only** if they fall into one of the categories defined below.

Every executable reference captured during compilation MUST be classifiable into exactly one category.

#### Category A — Expression-Representable References

An executable reference is **expression-representable** if it can be captured as syntax (e.g. AST) and recompiled or translated by the compiler.

Examples include:
- pure expressions
- arithmetic or logical predicates
- simple lambdas without free variables
- structural transformations

Expression-representable references:
- MUST be side-effect free
- MUST NOT capture external state
- MAY be re-targeted to multiple compilation backends

These references are preferred.

#### Category B — Runtime Dotted-Path References

An executable reference may be provided as, or resolved to, a **dotted-path reference** identifying a stable runtime symbol.

Examples:
- `"project.runtime.validators:is_valid_foo"`
- a function object resolvable to `(module, qualname)`

For such references:
- The compiler records only the dotted path, not the callable itself
- The referenced symbol MUST be importable at runtime
- The referenced module MUST NOT be an AML module
- The callable MUST NOT depend on AML-specific state

These references allow integration with conventional runtime Python code without coupling runtime execution to AML modules.

#### Category C — Runtime-Only References

Some executable references may be explicitly declared as **runtime-only**.

Runtime-only references:
- MAY appear in AML syntax
- ARE NOT guaranteed to be supported by all compilation targets
- MUST be explicitly marked as runtime-only

Compilation targets that do not support runtime-only references MUST either:
- ignore them, or
- raise a target-specific compilation error

Runtime-only references are intended for Python-specific behaviors such as debugging hooks, plotting helpers, or interactive affordances.

#### Disallowed References

Executable references that do not fall into Category A, B, or C are invalid.

In particular, AML declarations MUST NOT rely on:
- anonymous or non-resolvable callables
- closures capturing external or mutable state
- executable objects whose identity matters at runtime but cannot be relocated
- functions defined inline whose semantics depend on import context

Such references require importing AML modules at runtime and are therefore prohibited.

#### Validation

Conformance to the rules in this section is validated at **compile time**.

A compiler MAY:
- reject invalid references
- issue warnings for discouraged but detectable patterns
- offer a strict validation mode for CI or release workflows

Failure to comply with these rules results in undefined compilation behavior.

### Built-in functions

To ensure that temporal, math, string, and geometry functions behave symbolically inside AML declarations but evaluate normally at runtime, Anaximander uses a lightweight **symbolic evaluation context**. A thread-local flag (`_ctx.symbolic`) is enabled automatically by the **Prototype metaclass** when Python executes the body of an AML class during module import. As a result, calls like `nx.dt.now()` or `nx.dt.lag("5T")` inside field definitions or selection/view expressions return **DTI expression nodes** rather than real Python values. Outside AML—such as in normal Python code or default factories at runtime—the same functions evaluate eagerly and return standard Python types (e.g., `datetime`). This mechanism cleanly separates symbolic construction from runtime behavior without requiring special syntax, and it generalizes uniformly across all Anaximander namespaces (`nx.dt`, `nx.math`, `nx.str`, `nx.geo`).

### Prototype serialization

AML prototypes must have a canonical, fully serializable manifest representation. Every prototype and protodescriptor exposes a `to_manifest()` method that yields a pure data structure composed of primitives, symbolic references, and AML expression literals. Executable Python objects such as callables, parsers, validators, and reductions are never embedded directly; instead, their declarative identities are captured as strings or ASTs. When a manifest is reloaded, the AML loader recreates the prototype with identical declarative semantics, and the compiler reconstructs the executable runtime forms. This guarantees stability, portability, and compatibility with schema registries, versioning, DVS, and cross-environment compilers.

### Use of Generics

AML distinguishes two categories of generics. *Evaluation-type generics* (e.g., `Data[T]`, `Scalar[T]`) parameterize over Python runtime types (`int`, `str`, `Decimal`, …) and govern representation and parsing. *Prototype-type generics* (e.g., `List[T]`, `Interval[T]`) parameterize over AML prototypes and govern schema-level structure. The two must remain disjoint: evaluation-type parameters always refer to Python types, while prototype-type parameters always refer to AML prototypes or archetypes. This separation prevents semantic ambiguity and ensures correct behavior across parsing, validation, compilation, and runtime evaluation.

#### Special mention for FieldListDescriptor

```python
# Python 3.14+ (PEP 695 generics)

# Base mixin for descriptors that carry a list of fields
class FieldListDescriptor[F: FieldDescriptor](Protodescriptor):
    # Normalized, final list of field descriptors
    fields: tuple[F, ...]
    
    # Runtime whitelist used during normalization
    admissible_field_types: tuple[type[FieldDescriptor], ...] = (FieldDescriptor,)


# --- Concrete subclasses ------------------------------------------------------

class KeyDescriptor(FieldListDescriptor[DataDescriptor | LinkDescriptor]):
    admissible_field_types = (DataDescriptor, LinkDescriptor)


class SequenceDescriptor(FieldListDescriptor[DataDescriptor]):
    admissible_field_types = (DataDescriptor,)


class UnicityDescriptor(FieldListDescriptor[DataDescriptor | LinkDescriptor]):
    admissible_field_types = (DataDescriptor, LinkDescriptor)


class IndexDescriptor(FieldListDescriptor[DataDescriptor | LinkDescriptor]):
    admissible_field_types = (DataDescriptor, LinkDescriptor)


class PartitioningDescriptor(FieldListDescriptor[DataDescriptor | LinkDescriptor]):
    admissible_field_types = (DataDescriptor, LinkDescriptor)


class FieldGroupDescriptor(FieldListDescriptor[FieldDescriptor]):
    # Field groups accept any field descriptor defined on the model,
    # and flatten nested groups into concrete field descriptors.
    admissible_field_types = (FieldDescriptor,)


```

### Field Expressions

For callable field expressions, compilers may obtain the function’s source code (where available), parse it into a Python AST, and restrict themselves to the function body’s return expression (or a single-expression body). This AST is then validated against the allowed expression subset and translated into an internal expression tree that is backend-agnostic. Backends compile this intermediate representation to their native form (e.g. SQL, Ibis). If source code is unavailable, or the AST contains unsupported constructs, the descriptor is marked as runtime-only and the callable is invoked directly at evaluation time. For string-based expressions, a simpler parser can be used (e.g. for dotted paths or a small expression grammar), either directly constructing the same internal expression tree or resolving them to previously declared attributes.
