# AML Diagnostics Mapping (Draft)

This document inventories current AML exceptions and maps them to proposed diagnostic classes.
It is intended as a refactor guide for replacing ad-hoc exceptions with structured diagnostics.

## Conventions

- **Severity**: `error` for user-facing validation failures; `fatal` for internal invariant violations.
- **Context**: expected diagnostic bag scope (`project`, `module`, `prototype`, `declarator`).
- **Templates**: prefer class-level templates with kwargs for recurring messages; use `repr()`
  for declarators/prototypes where helpful.

## Taxonomy (Proposed)

### Namespace & Declaration (error)
- `DeclarativeNamespaceRedefinition`
- `DeclarativeNamespaceDeletion`
- `DeclaratorNameInvalid`
- `DeclaratorNameConflict`
- `BindingAlreadyRegistered`
- `UnnamedDeclaratorRegistration`

### Declarator Configuration & Overrides (error)
- `DeclaratorHandleInvalid`
- `DeclaratorHandleConflict`
- `DeclaratorOverrideViolation`
- `DeclaratorReassignmentViolation`
- `DeclaratorTypeConstraintViolation`
- `DeclaratorNullabilityViolation`

### Prototype / Archetype / Trait Semantics (error)
- `InvalidPrototypeBase`
- `MultipleInheritanceUnsupported`
- `InvalidTypeRole`
- `TraitConformanceViolation`
- `ArchetypeInheritanceViolation`

### Annotation & Type Resolution (error)
- `AnnotationMissing`
- `AnnotationInvalid`
- `TypeHintResolutionFailed`

### Module / AST Constraints (error)
- `DisallowedModuleStatement`
- `InvalidModuleSource`

### Reference Resolution (error)
- `ReferenceInvalid`
- `ReferenceKindInvalid`
- `ReferenceMissingOwnerMember`

### Registry Errors (error)
- `RegistryKeyMissing`
- `RegistryHandleUnknown`
- `RegistryTypeAmbiguous`
- `RegistryTypeUnknown`
- `RegistryConflict`
- `RegistryArgumentInvalid`

### Binding / Validation (error)
- `BindingInvalidValue`
- `BindingTypeMismatch`
- `ValidatorFailed`
- `EnumerationInvalid`

### Internal Invariants (fatal)
- `DeclaratorInvariantViolation`
- `DiagnosticConfigurationError`
- `RegistryInvariantViolation`

## Inventory & Mapping

### `src/anaximander/aml/declarative.py`
- L81: "Cannot redefine name '{key}' in declarative namespace."
  - `DeclarativeNamespaceRedefinition` (error, declarator)
- L90: "Cannot delete items from a declarative namespace."
  - `DeclarativeNamespaceDeletion` (error, declarator)
- L103: "Expected a Declarator instance, got {declarator}."
  - `DeclaratorTypeConstraintViolation` (error, declarator)
- L107: "Declarator names cannot contain '.'."
  - `DeclaratorNameInvalid` (error, declarator)
- L116: Duplicate declaration msg
  - `DeclaratorNameConflict` (error, declarator)
- L134: "Binding '{key}' is already registered in with handle '{handle}'."
  - `BindingAlreadyRegistered` (error, declarator)
- L138: "Binding '{key}' is already registered in this namespace."
  - `BindingAlreadyRegistered` (error, declarator)
- L152: "Name '{key}' is neither a declaration nor a binding in strict mode."
  - `DeclarativeNamespaceRedefinition` (error, declarator)
- L217: "Unnamed declarator registered outside class assignment."
  - `UnnamedDeclaratorRegistration` (error, declarator)

### `src/anaximander/aml/declarators.py`
- L120: "MISSING has no truth value"
  - `DeclaratorInvariantViolation` (fatal, declarator)
- L359: "Declarator must be bound to a class before accessing its key."
  - `DeclaratorInvariantViolation` (fatal, declarator)
- L408: "__reserved_patterns__ validation"
  - `DeclaratorTypeConstraintViolation` (error, declarator)
- L416: "Declarator subclasses cannot define an empty __handle__ ..."
  - `DeclaratorHandleInvalid` (error, declarator)
- L420: "Declarator handle '{cls.__handle__}' is already registered."
  - `DeclaratorHandleConflict` (error, declarator)
- L435: "Declarator handle collision in mro"
  - `DeclaratorHandleConflict` (error, declarator)
- L449 / L458: `_validate_value_type` failures
  - `DeclaratorTypeConstraintViolation` (error, declarator)
- L478 / L481: "Cannot use reserved name ..."
  - `DeclaratorNameInvalid` (error, declarator)
- L514: "Declarator ... cannot be bound to a value."
  - `DeclaratorOverrideViolation` (error, declarator)
- L527: "Declarator ... cannot be overridden."
  - `DeclaratorOverrideViolation` (error, declarator)
- L658: "EnumerationDeclarator.members must be a tuple of strings."
  - `DeclaratorTypeConstraintViolation` (error, declarator)
- L671: "EnumerationDeclarator.members length mismatch"
  - `DeclaratorTypeConstraintViolation` (error, declarator)
- L721 / L774 / L803: "Unsupported binding type"
  - `BindingTypeMismatch` (error, declarator)
- L728 / L781 / L1095: "Non-nullable ... cannot be bound to None."
  - `DeclaratorNullabilityViolation` (error, declarator)
- L739 / L744 / L752+ (override/loosen)
  - `DeclaratorOverrideViolation` (error, declarator)
- L808 / L1056: "Field cannot be reassigned."
  - `DeclaratorReassignmentViolation` (error, declarator)
- L940 / L1271 / L1362 / L1513+ (type constraint violations)
  - `DeclaratorTypeConstraintViolation` (error, declarator)
- L1006 / L1382 / L1510 (members required)
  - `EnumerationInvalid` (error, declarator)
- L1121 / L1158: "Link/Backlink fields are not prototype-bindable."
  - `DeclaratorOverrideViolation` (error, declarator)

### `src/anaximander/aml/prototype.py`
- L290 / L292: "Expected a Trait/Archetype type ..."
  - `DeclaratorTypeConstraintViolation` (error, prototype)
- L319 / L479: "Archetypes, traits and prototypes cannot be instantiated directly."
  - `InvalidPrototypeBase` (error, prototype)
- L366: "Base class ... is not a valid AML prototype base."
  - `InvalidPrototypeBase` (error, prototype)
- L368: "Prototypes do not support multiple inheritance."
  - `MultipleInheritanceUnsupported` (error, prototype)
- L398 / L433 / L502: "Declarator not allowed / metadata not declared"
  - `DeclaratorTypeConstraintViolation` (error, prototype)
- L427: "Metadata ... not declared"
  - `BindingInvalidValue` (error, prototype)
- L536: "Only trait types have a supertrait."
  - `TraitConformanceViolation` (error, prototype)
- L548: "Expected an Archetype or Trait type ..."
  - `DeclaratorTypeConstraintViolation` (error, prototype)
- L577 / L597 / L618: invalid view values
  - `DeclaratorTypeConstraintViolation` (error, prototype)
- L644: "Annotatable declarators must be named before annotation binding."
  - `AnnotationInvalid` (error, prototype)
- L652: "Missing annotation for declarator '{name}'."
  - `AnnotationMissing` (error, prototype)

### `src/anaximander/aml/modules.py`
- L181 / L188: "Reference must include owner and member."
  - `ReferenceMissingOwnerMember` (error, module)
- L199 / L201: "Unknown type reference / reference kind."
  - `ReferenceKindInvalid` (error, module)
- L217: "Invalid reference '{body}'."
  - `ReferenceInvalid` (error, module)
- L240: "Expected prototype reference, got ..."
  - `ReferenceInvalid` (error, module)
- L315 / L318 / L322: "AML disallows statement ..."
  - `DisallowedModuleStatement` (error, module)
- L346: "Module ... has no file origin to parse."
  - `InvalidModuleSource` (error, module)
- L431: "Failed to resolve type hints ..."
  - `TypeHintResolutionFailed` (error, module)
- L455 / L457 / L461 / L465 / L473 / L475: role validation
  - `InvalidTypeRole` (error, module)
- L501+ / L519+ / L539+ / L546+ / L555+ / L568+ / L575+ / L581+ / L588+ / L594+ / L664+
  - `BindingInvalidValue` or `EnumerationInvalid` (error, module)

### `src/anaximander/aml/registries.py`
- L73: "Key {key} not found in registry."
  - `RegistryKeyMissing` (error, module)
- L196: "Module {key} is already registered."
  - `RegistryConflict` (error, module)
- L223: "Type {type_} does not provide a declarative key."
  - `RegistryArgumentInvalid` (error, module)
- L248 / L255 / L258: Unknown or ambiguous types
  - `RegistryTypeUnknown` / `RegistryTypeAmbiguous` (error, module)
- L389 / L490 / L502: binding/override failures
  - `RegistryInvariantViolation` (fatal, module) or `RegistryConflict` (error)
- L415 / L417 / L420: invalid registry arguments
  - `RegistryArgumentInvalid` (error, module)
- L494 / L537 / L611 / L818: missing keys/registries
  - `RegistryKeyMissing` / `RegistryHandleUnknown` (error, module)

### `src/anaximander/aml/handles.py`
- L88: "decorator cannot mix members and callable."
  - `DeclaratorTypeConstraintViolation` (error, declarator)
- L129: "Invalid handle '{cls.__handle__}' for declarator handle."
  - `DeclaratorHandleInvalid` (error, declarator)
- L152: "handle can only be used in declarative bodies."
  - `DeclarativeNamespaceRedefinition` (error, declarator)
- L180: "handle does not support item assignment."
  - `DeclaratorOverrideViolation` (error, declarator)
- L236 / L417: "handle does not support validators."
  - `DeclaratorTypeConstraintViolation` (error, declarator)

### `src/anaximander/aml/model.py`
- L82 / L93 / L160 / L172: nxfield binding errors
  - `BindingTypeMismatch` (error, prototype)
- L84: missing field keys
  - `BindingInvalidValue` (error, prototype)

### `src/anaximander/aml/archetype.py`
- L49: "Expected a prototype instance, got ..."
  - `InvalidPrototypeBase` (error, prototype)
- L51: "Archetype ... must directly inherit from another archetype."
  - `ArchetypeInheritanceViolation` (error, prototype)

### `src/anaximander/aml/trait.py`
- L50: "Expected a prototype instance, got ..."
  - `InvalidPrototypeBase` (error, prototype)
- L52: "Trait ... must directly inherit from an archetype or trait."
  - `TraitConformanceViolation` (error, prototype)
- L56: "Traits cannot declare or inherit protodescriptors."
  - `TraitConformanceViolation` (error, prototype)

### `src/anaximander/aml/diagnostics.py`
- L125: "Diagnostic has neither message nor template"
  - `DiagnosticConfigurationError` (fatal, module)
