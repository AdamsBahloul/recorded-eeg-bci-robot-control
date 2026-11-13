# GitHub publishing checklist

The repository is structured for source control, but review these items before making it public:

1. **Choose a license.** No license has been selected in this repository. Add the license you intend to grant; do not assume a default.
2. **Review staged files.** Confirm there are no EEG recordings, processed epochs, model binaries, deployment bundles, private addresses, or session logs in the commit. `.gitignore` excludes `data/raw/`, `data/processed/`, caches, and local environments.
3. **Document data access.** Keep the dataset source and its terms clear. Do not mirror the EEG recordings in the repository.
4. **Verify the setup guide.** The Pi deployment guide targets 64-bit Raspberry Pi OS and Python 3.11. Mark hardware checks as unverified unless they have been run on the physical device.
5. **Review current project state.** `git status --short` may include earlier local implementation changes. Inspect and commit the intended source and documentation together; do not assume all current changes belong to this documentation cleanup.

Publishing means creating a GitHub repository and pushing commits. Review the repository status and staged file list before each release.
