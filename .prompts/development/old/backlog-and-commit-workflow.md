# Backlog & Commit Message Workflow - AI Reference

This workflow is for updating documentation (CHANGELOG.md, backlog files) and generating commit messages after implementing features.

## When to Use

Apply this workflow when:
- User asks to "update the backlog"
- User requests a commit message
- User says "repeat the steps" or "do the same as before"
- After completing a feature implementation

## Workflow Steps

### Step 1: Review Recent Commits

```bash
# Get last 5 commits with full info
git log --oneline -5

# Get detailed commit info with stats
git show <commit-hash> --stat

# Check commits since specific date
git log --oneline --since="YYYY-MM-DD"

# View commit with day of week
git log --date=format:'%Y-%m-%d %H:%M:%S %a' --pretty=format:'%h - %ad - %s' -10
```

**Goal**: Understand what was implemented in recent commits

### Step 2: Check Current Git Status

```bash
# Check current status
git status

# Check staged changes
git diff --cached --stat

# Check unstaged changes
git diff --stat

# List untracked files
git status --short
```

**Goal**: Identify what changes are ready to commit

### Step 3: Read Documentation Files

Read these files to understand current state:
- `CHANGELOG.md` (top 50 lines usually sufficient)
- `docs/backlog/infrastructure.md`
- `docs/backlog/video-streaming.md`
- `docs/backlog/kernels.md`
- `docs/backlog/neural-networks.md`
- `docs/backlog/README.md`

**Goal**: Know what's already documented and what needs updating

### Step 4: Analyze Changes

Based on commits and staged files, identify:
1. **What was implemented** (new features, tests, configurations)
2. **What backlog items are now complete** (mark with `[x]`)
3. **What new sections to add** to CHANGELOG.md
4. **Technology/patterns used** (e.g., Playwright, BDD, Proto changes)

**Key patterns to look for**:
- New test files → Testing section updates
- Proto changes → API updates
- New features → Feature sections
- Config changes → Configuration updates
- Docker changes → Infrastructure updates

### Step 5: Update CHANGELOG.md

**Format**: Add new section at the top of October 2025 section

```markdown
### [Feature Name] (Oct XX, 2025)
- [x] Specific accomplishment with details
- [x] Another accomplishment
- [x] Include numbers when relevant (X tests, Y lines, Z scenarios)
- [x] Mention technologies used
- [x] Note integration points (Docker, CI/CD, etc.)
```

**Rules**:
- Use checkboxes `[x]` for completed items
- Be specific with numbers (tests count, lines of code, scenarios)
- Group related items logically
- Mention all affected components
- Include validation status (e.g., "All tests passing")

### Step 6: Update Backlog Files

**For `docs/backlog/infrastructure.md`**:

Mark completed items:
```markdown
- [x] Completed task description
```

Add new subsections if needed:
```markdown
### New Category Name
- [x] Completed item
- [x] Another completed item
- [ ] Future task
- [ ] Another future task
```

Update scenario counts:
```markdown
- [x] 37 scenarios passing (update from previous count)
```

**For other backlog files**: Similar pattern based on content

### Step 7: Stage Documentation Changes

```bash
# Stage specific documentation files
git add CHANGELOG.md docs/backlog/infrastructure.md

# Or stage all changes if appropriate
git add -A
```

### Step 8: Generate Commit Message

**Format**: Conventional Commits

```
<type>: <description>

- Bullet point 1
- Bullet point 2
- Bullet point 3
- Update changelog and backlog

[Optional footer for breaking changes]
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `test`: Adding tests
- `refactor`: Code refactoring
- `chore`: Maintenance tasks
- `perf`: Performance improvements
- `ci`: CI/CD changes
- `build`: Build system changes

**Description rules**:
- Lowercase, imperative mood
- No period at end
- Max 72 characters for first line
- Be specific about what was added/changed

**Examples**:

```bash
# Testing feature
test: add playwright e2e testing with 9 comprehensive suites

- Add 9 test suites with 37 E2E tests (954 lines)
- Create test helpers and utilities (182 lines)
- Add data-testid attributes to all components
- Configure Playwright for Chrome, Firefox, and WebKit
- Integrate with Docker for CI/CD pipeline
- Update changelog and infrastructure backlog
```

```bash
# API change (breaking)
feat!: add api 2.0.0 with dynamic filter metadata discovery

- Add FilterDefinition and FilterParameter protobuf messages
- Enable dynamic discovery of filter parameters
- Rename ACCELERATOR_TYPE_GPU to ACCELERATOR_TYPE_CUDA
- Implement processor_capabilities.feature with 4 scenarios
- Update changelog and infrastructure backlog

BREAKING CHANGE: API upgraded to 2.0.0 with new capability structure
```

```bash
# Documentation only
docs: update changelog and backlog for oct 15-16 work

- Add changelog entries for multi-source video grid and BDD testing
- Mark completed items in infrastructure and video-streaming backlogs
- Document 29 passing BDD scenarios and critical bug fixes
```

### Step 9: Provide Commit Message to User

Provide TWO versions:
1. **Full version** with all details
2. **Short version** for quick commits

Example:
```
Full version:
test: add playwright e2e testing with 9 comprehensive suites
[... full details ...]

Or short version:
test: add playwright e2e testing with 9 comprehensive suites

- Add 9 test suites with 37 tests
- Update changelog and backlog
```

---

## Quick Checklist

When user requests backlog update:

1. [ ] Run `git log --oneline -5` and `git show` for recent commits
2. [ ] Run `git status` and `git diff --cached --stat`
3. [ ] Read CHANGELOG.md (top section)
4. [ ] Read relevant backlog files
5. [ ] Identify what was implemented
6. [ ] Update CHANGELOG.md with new section
7. [ ] Update backlog files (mark completed, update counts)
8. [ ] Stage changes: `git add CHANGELOG.md docs/backlog/*`
9. [ ] Generate commit message (full + short versions)
10. [ ] Present commit message to user

---

## Common Patterns

### Pattern 1: New Testing Framework
- Add section to CHANGELOG.md under Testing
- Update `infrastructure.md` → Load Testing & BDD section
- Include test counts, file counts, line counts
- Mention CI/CD integration if applicable

### Pattern 2: API Changes
- Add section to CHANGELOG.md under API/Architecture
- Update `infrastructure.md` → Infrastructure section
- Note version bumps
- Document breaking changes clearly

### Pattern 3: New Features
- Add section to CHANGELOG.md with date
- Mark related backlog items as complete
- Add new subsections if feature enables future work
- Update scenario/test counts

### Pattern 4: Configuration Changes
- Mention in relevant CHANGELOG section
- Update infrastructure backlog if affects deployment
- Note environment variables or config file changes

---

## Examples from Project

### Good Changelog Entry
```markdown
### End-to-End Testing with Playwright (Oct 17, 2025)
- [x] Integrate Playwright testing framework with TypeScript
- [x] Add 9 comprehensive E2E test suites (954 lines of tests)
- [x] Implement drawer-functionality tests (82 lines, 4 tests)
- [x] Configure Playwright for Chrome, Firefox, and WebKit
- [x] Create run-e2e-tests.sh script for automated test execution
- [x] Add Playwright service to docker-compose.dev.yml
- [x] All tests passing with cross-browser validation
```

### Good Backlog Update
```markdown
### Playwright E2E Tests (Frontend)
- [x] Setup Playwright with TypeScript
- [x] Configure multi-browser testing (Chrome, Firefox, WebKit)
- [x] Test: Drawer functionality (4 tests)
- [x] Test: Filter configuration (5 tests)
- [x] Create reusable test helpers
- [x] Docker integration for CI/CD
- [ ] Visual regression testing with screenshots
- [ ] Performance testing with Lighthouse
```

### Good Commit Message
```
test: add playwright e2e testing framework with 9 comprehensive suites

- Integrate Playwright with TypeScript for multi-browser testing
- Add 9 E2E test suites covering all UI functionality (954 lines)
- Create test-helpers utility with 182 lines of reusable functions
- Add data-testid attributes to all interactive components
- Configure test artifacts and HTML reports
- Add Playwright service to docker-compose.dev.yml
- Update changelog and infrastructure backlog

All tests passing with cross-browser validation (Chrome, Firefox, WebKit)
```

---

## Notes

- Always read actual file contents, don't assume
- Check git history to understand what changed
- Be specific with numbers and metrics
- Group related changes logically
- Keep commit messages under 72 chars for first line
- Update ALL relevant backlog files, not just one
- Mark items as complete `[x]` only if fully done
- Add new pending items `[ ]` if feature enables future work

