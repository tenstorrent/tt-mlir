# Specifications

Specifications are documents that define the requirements for features or
concepts that are particularly cross-cutting, complex, or require a high degree
of coordination and planning. They are intended to be a living document that
evolves as the feature is developed and should be maintained as the goto
reference documentation for the feature or concept.

Specifications are written in markdown and are stored in the `docs/src/specs`
directory of the repository. Below is a template that should be used when
creating a new specification.

## Specification Template

```markdown
# [Title]

A brief description of the feature or concept that this specification is
defining.

## Motivation

A description of why this feature or concept is needed and what problem it is
solving. This section is best written by providing concrete examples and use
cases.

## Proposed Changes

A list of the components that will be impacted by this spec and a detailed
description of the changes that will be made to each respective component.

It should also call out any interactions between components and how they might
share an interface or communicate with each other.

## Test Plan

A brief description of how the feature or concept will be tested.

## Concerns

A list of concerns that have been identified during the design of this feature.
```
