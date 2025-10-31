git branch# Contributing Guide (mAI Coach)

How to set up, code, test, review, and release so contributions meet our **Definition of Done (DoD)** and pass CI. This guide mirrors our Team Charter and makes the quality gates explicit.

---

## Code of Conduct

We follow the Contributor Covenant v2.1.
To report unacceptable behavior, email **whitnetr@oregonstate.edu** and **seiferco@oregonstate.edu**.

**Owner:** Cole Seifert • **Next review:** Nov 15, 2025

---

## Getting Started

**Prerequisites**
- macOS with **Xcode 16+** (iOS 18 Simulator).
- Homebrew (optional): `brew install swiftlint swiftformat`.
- Git + GitHub access to the repository.

**Clone & open**
```bash
git clone https://github.com/TWhit229/mAI_Coach.git
cd mAI_Coach
open mAICoach/mAICoach.xcworkspace
```

**Environment / Secrets**
- Do **not** commit secrets or API keys.
- Use `Config.xcconfig` (ignored in VCS) for secrets or per‑dev settings; commit `Config.xcconfig.example` with placeholders.
- If a secret is committed, rotate it immediately and open a PR removing it.

**Run the app locally**
- In Xcode, select **iPhone 15 (iOS 18)** simulator -> **Run**.
- If SPM packages act up: **File -> Packages -> Reset Package Caches**, then Clean Build Folder.

**Owner:** Travis Whitney • **Next review:** Nov 15, 2025

---

## Branching & Workflow

We use **trunk‑based development** on `main` with short‑lived branches.

- **Branch names:** `feat/<short-slug>`, `fix/<short-slug>`, `chore/<short-slug>`, `docs/<short-slug>`
- **Rebase before PR:** `git fetch && git rebase origin/main`
- **Small PRs:** target < ~400 LOC net change
- **Merging:** squash‑and‑merge after approvals and green checks

**Owner:** Cole Seifert • **Next review:** Nov 15, 2025

---

## Issues & Planning

- Open an issue for every change. Use the issue template and one clear outcome.
- **Labels:** `feature`, `bug`, `chore`, `docs`, `infra`, `research`
- **Estimate:** S (≤½ day) / M (≤1 day) / L (>1 day)
- **Assignment & triage:** during Tues/Thurs work sessions

**Owner:** Cole Seifert • **Next review:** Nov 15, 2025

---

## Commit Messages

Follow **Conventional Commits**.

**Examples**
- `feat(overlay): draw joint landmarks on video`
- `fix(export): correct timestamp rounding`
- `chore(ci): add SwiftFormat check`
- `docs(readme): add setup steps`

Reference issues in the footer (auto‑close on merge): `Closes #123`

**Owner:** Travis Whitney • **Next review:** Nov 15, 2025

---

## Code Style, Linting & Formatting

- **Swift style:** enforced by **SwiftLint** and **SwiftFormat**.
- **Config files:** `.swiftlint.yml` and `.swiftformat` at repo root.
- **Local checks**
```bash
swiftlint
swiftformat . --lint     # check
swiftformat .            # apply fixes
```

A PR is merge‑ready only if it’s **lint‑clean and format‑clean** locally and in CI.

**Owner:** Travis Whitney • **Next review:** Nov 29, 2025

---

## Testing

**What to test**
- Unit tests for logic (exporter/parser, overlay math, thresholding/smoothing)
- Add/adjust tests when you change logic, fix a bug, or add a feature

**How to run (CLI)**
```bash
xcodebuild   -scheme mAICoach   -destination 'platform=iOS Simulator,name=iPhone 15'   clean test
```

**Coverage**
- No numeric floor yet, but **new/changed logic must be covered**. Document any gaps and file a follow‑up issue.

**Owner:** Travis Whitney • **Next review:** Nov 29, 2025

---

## Pull Requests & Reviews

**Before you open a PR**
- Rebase on latest `main`
- Local build & tests pass; lint/format pass
- Update docs if behavior/usage changes
- Link the issue and add a brief **How to test**

**PR requirements**
- **1 approving review** from the other teammate (review within ~24 hrs when possible)
- All required checks green (see **CI/CD**)

**Reviewer focus**
- Correctness, clarity, scope, maintainability, and performance
- Request changes for correctness or DoD violations; prefer suggestions for style

**Owner:** Cole Seifert • **Next review:** Nov 15, 2025

---

## CI/CD

**Provider:** GitHub Actions (`.github/workflows/ios.yml`)

**Required jobs (must pass before merge)**
- `build-and-test` — `xcodebuild + XCTest` (iOS 18 sim)
- `lint` — `swiftlint`
- `format-check` — `swiftformat --lint`

**Viewing & re‑running**
- Repo -> **Actions** -> pick the run -> job -> logs
- Re‑run failed jobs from the run page

**Branch protection**
- `main` is protected by the jobs above

**Owner:** Travis Whitney • **Next review:** Nov 29, 2025

---

## Security & Secrets

- **No hard‑coded secrets** (tokens, API keys) in code or Info.plist
- Use Keychain and/or `Config.xcconfig` (ignored) for sensitive values; commit `Config.xcconfig.example`
- Rotate any exposed secret immediately; document rotation in the PR
- Keep dependencies reasonably current; prioritize fixes for security advisories

**Owner:** Cole Seifert • **Next review:** Nov 22, 2025

---

## Documentation Expectations

Update docs whenever behavior, flags, or usage changes:
- `README.md`
- `/docs/` pages (incl. meeting notes template)
- API doc comments for public types/functions
- `CHANGELOG.md` (keep a short, dated list of notable changes)

**Owner:** Cole Seifert • **Next review:** Nov 15, 2025

---

## Release Process

Fall term uses **milestone tags** (no App Store deploy yet).

1) Bump Marketing Version in Xcode target  
2) Update `CHANGELOG.md`  
3) Tag and push:
```bash
git tag v0.1.0 -m "Fall demo baseline"
git push origin v0.1.0
```
4) If TestFlight is added later, document archive/export steps and link Apple docs

**Owner:** Travis Whitney • **Next review:** Nov 29, 2025

---

## Support & Contact

- **Issues first:** open a GitHub issue with repro steps (attach a short clip if relevant)
- **Maintainers**
  - Travis Whitney — whitnetr@oregonstate.edu
  - Cole Seifert — seiferco@oregonstate.edu
- **Response window:** 6 hours (09:00–21:00 PT) or within 24 hours
- **Escalation:** bring blockers to next **Instructor** meeting (Wed 1300 with Alex Ulbrich) or **TA** meeting (Fri 1500 with Nischal Aryal)

**Owner:** Cole Seifert • **Next review:** Nov 15, 2025

---

### Cross‑References

- **Team Charter -> DoD & Quality Gates** lists the named CI jobs and checks enforced by this guide.
