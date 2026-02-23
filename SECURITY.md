# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 1.0.x   | Yes                |
| < 1.0   | No                 |

## Reporting a Vulnerability

If you discover a security vulnerability in Endgame, please report it responsibly.

**Do not open a public GitHub issue for security vulnerabilities.**

Instead, please use one of the following methods:

1. **GitHub Security Advisories** (preferred): Go to the [Security tab](https://github.com/allianceai/endgame/security/advisories) and click "Report a vulnerability"
2. **Email**: Send details to the maintainers via the email listed in the GitHub organization profile

### What to include

- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

### What to expect

- Acknowledgment within 48 hours
- Status update within 7 days
- We aim to release a fix within 30 days for confirmed vulnerabilities

## Scope

Security issues in the following areas are in scope:

- Code execution vulnerabilities in model loading/deserialization (`endgame.persistence`)
- ONNX export producing models with unintended behavior
- Dependency vulnerabilities in core dependencies
- Path traversal or injection in file-handling utilities

## Out of Scope

- Adversarial ML attacks on trained models (this is a research area, not a software vulnerability)
- Denial of service through large inputs (expected behavior for ML workloads)
- Issues in optional dependencies not maintained by this project
