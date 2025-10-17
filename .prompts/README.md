# AI Prompts Library

This directory contains AI prompts used for testing, development, and maintenance of the CUDA Image Processor project.

## Directory Structure

```
.prompts/
├── README.md              # This file
├── testing/               # Testing and QA prompts
│   └── manual-testing-*   # Manual testing procedures
└── development/           # Development and architecture prompts
    └── ...                # Future prompts
```

## Naming Convention

Prompts follow this naming pattern:
- `{purpose}-{feature}-{area}.md`
- Example: `manual-testing-multi-source-grid.md`

## Usage

1. **Testing Prompts**: Use these prompts to validate features and ensure quality
2. **Development Prompts**: Use these to guide development tasks and architecture decisions
3. **Maintenance**: Keep prompts updated when features change

## Version Control

- All prompts are version controlled via Git
- Update prompts when related features change
- Document major changes in commit messages

## Adding New Prompts

1. Choose appropriate directory (`testing/`, `development/`, etc.)
2. Follow naming convention
3. Include clear success criteria
4. Add metadata section (date, version, author)
5. Commit with descriptive message

## Related Documentation

- See `/docs` for general project documentation
- See `/integration/tests/acceptance` for BDD test specifications
- See `CHANGELOG.md` for feature history

