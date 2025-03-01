# Contributing guidelines for TT-Forge

## PR Guidelines

### Community contributions
Thank you for your interest in the TT-Forge project, we appreciate your support.
For all PRs, we have an internal policy listed below which your PR will go through after an initial review has been done.

The initial review will encompass the following:
* Reviewing the PR for CI/CD readiness, making sure that the code and PR at a high level make sense for the project.
* Once approved for CI/CD readiness, a Tenstorrent developer will kick off our CI/CD pipeline on your behalf.

### Internal contributions
For internal contributions we have the following guidelines:

* A 24 hour merge rule exists. The rule is to wait at least 24 hours since the PR was initially opened for review. This gives members of our teams that span the globe opportunity to provide feedback to PRs.

In addition to the 24 hour rule, the following prerequisites for landing a PR exist:
* At least 1 reviewer signs off on the change
* Component owners sign-offs (GitHub will tell you if this hasn't been met)
* Green CI
* Wait at least 24 hours since opening the PR to give all tagged reviewers a chance to take a look, or at least comment on the issue that they need more time to review.
  * *Rebasing or further changes to the PR do not reset the 24 hour counter.*

### Coding Guidelines and Standards
Before submitting your PR, ensure your code adheres to the project’s [Coding Guidelines and Standards](./docs/src/coding-guidelines.md). These guidelines outline expectations for code style, formatting, and best practices to maintain consistency and quality across the project.
