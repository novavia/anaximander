# Copilot Instructions

## Sources of Truth

- `README.md` is currently a placeholder.
- `ANAXIMANDER.md` describes the goals and key tenets of the project.
- `docs/implementation/IMPLEMENTATION-PLAN.md` describes implementation guidelines and conventions.
- Design documents under `docs/` describe the project's architecture, semantics, and design decisions.

When generating or editing code, align with these documents and do not
introduce concepts or terminology that contradict them.

## Style and Formatting

- **High Density**: Prefer high-density code and one-liners for simple assignments, function calls, and collection literal definitions.
- **Avoid Vertical Expansion**: Do not "explode" lists, dictionaries, or function arguments into multiple lines unless they exceed the 99-character limit.
- **Ruff Alignment**: Align with the project's Ruff configuration (`line-length = 99`, `skip-magic-trailing-comma = true`).
- **Concatenation**: Avoid unnecessary line breaks between related logic blocks or decorators.
- **Type Annotations**: Use built-in types for annotations (e.g., `list` instead of `List` from `typing`).
- **Docstrings**: Use triple double-quoted (`"""`) docstrings for all public classes, methods, and functions. Please refer to the docstring conventions in `docs/implementation/IMPLEMENTATION-PLAN.md`.
- **Section Comments**: Use section comments (e.g., `# -------------------------`) to delineate logical sections within classes or modules.
- **Consistent Naming**: Follow existing naming conventions for functions, methods, variables, and classes.
- **Error Handling**: Use specific exception types and provide informative error messages.
- **Data Structures**: Use appropriate data structures for the task, ensuring clarity and efficiency.
- **Method Definitions**: Define methods clearly, ensuring parameters and return types are well-specified.
- **Code Clarity**: Prioritize code clarity and maintainability, ensuring that the code is easy to read and understand.
- **Avoid Redundancy**: Do not include redundant or unnecessary code constructs.