"""Initialization of AML's core package.

The core package contains the fundamental building blocks of the Anaximander Modeling Language (AML). It is designed as
functionally independent from the definition of the language itself. Instead, the core package provides infrastructure
aimed at writing a domain specific language that makes the following assumptions:
- The language is declarative in nature. Its declaration units are Python classes called prototypes. Prototypes can never
be instantiated directly, but they serve as models for runtime classes that are created through either compilation or
metaprogramming.
- Prototypes are organized in a type hierarchy. That hierarchy is comprised of archetypes, traits and genuine prototypes. 
Archetypes define the structure and behavior of the prototypes that inherit from them, while traits provide additional modular 
functionality that can be mixed into prototypes. Genuine prototypes are the concrete models that can be compiled or transformed 
into runtime classes. A prototype must inherit from exactly one archetype as its first base, either direclty or through
inheritance from a parent prototype, along with zero or more traits. The traits themselves must be compatible with the base archetype,
i.e. they must inherit from the same archetype or one of its bases. Archetypes can also define metacharacters, which then become 
class keyword arguments for all prototypes that inherit from them (i.e. similar to setting kwargs on __init_subclass__).
- Prototypes, archetyes and traits are all defined as Python classes, and share a common metaclass called prototype. The body of 
protoype, archetype or type classes is made up of declarative statements, in effect attributes, properties and methods, using a special 
set of declarative objects called declarators. Declarators define many features, spanning how they are declared in a class body, 
how they are validated, whether and how they can be overridden in derived prototypes, whether and how they can be bound to a set value, etc. 
Besides, they can be organized in a class hierarchy that reflects their semantic meaning in the language definition, which in turns conditions 
how they are interpreted and compiled into runtime and system code. The declarators are split into two categories: protodescriptors and 
metadescriptors. Protodescriptors are used to declare attributes and methods in prototypes. In particular, protodescriptors belong to 
the semantic domain of the application that is being modeled with prototypes. Metadescriptors span all other aspects, including class-level 
metadata described in archetypes and traits, system attributes, for instance schema indexes or runtime options, and any other non-domain-specific 
construct.
- The body of prototypes consists of protodescriptor statements. Protodescriptors declare attributes very much like Python descriptors,
but they do not carry the descriptor protocol since prototypes are not instantiated directly. Instead, protodescriptors serve to collect 
metadata about the attributes they declare, which gets compiled into attribute definitions in system code libraries.
- Archetypes and traits are declared using decorators. These decorators relax inheritance constraints in order to allow injection of
built-in or library classes in the inheritance hierarchy. The purpose of doing so is to instruct type checkers to treat derived
prototypes as instances of those built-in or library classes, which greatly facilitates domain model development by providing the
benefits of static type checking and IDE auto-completion. The core package provides the archetype and trait decorators for that purpose.
The body of archetypes and traits is populated with metadescriptors. Like protodescriptors, these are declarator classes that do not
implement the descriptor protocol, but instead collect metadata about the attributes they declare. 
- Declarators can optionally be organized into hierarchical namespaces. Namespaces serve to group related declarators together, and isolate
subdomain to avoid naming conflicts. In the Anaximander framework, the key namespace is of course 'nx', which serves as the interface
between the application domain and the system domain. To establish consistency with Python syntax, namespace declarators must be defined
and colleced in an inner class. Further, if these declarators are to be bound in prototype declarations (as is the case for class-level
metadata), the value assignments must also be nested -however in this instance, using a collection attribute as a class variable to serve as
the declarative container, or as is the case with metacharacters, in the class header. The core package provides basic namespace management 
functionality to facilitate the definition and usage of namespaces.
- The core package also provides compilation infrastructure that allows prototypes to be transformed into runtime classes. This includes
collection of declarators by the prototype metaclass, as well as filtered and structured AST representations of prototype bodies. However,
the actual compilation process is left to the specific language implementation, since it is highly domain-specific.
- Finally, the core package provides basic error handling and reporting mechanisms to facilitate debugging and development of domain models.
"""