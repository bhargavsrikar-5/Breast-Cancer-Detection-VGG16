# Security and sensitive data

Do not commit credentials, access tokens, private keys, `.env` files, or other secrets. Revoke and rotate any secret that may have been exposed; removing it in a later commit does not erase it from Git history.

Do not publish BUSI or BUS-BRA scans, masks, patient identifiers, personal health information, or screenshots containing medical data. Keep datasets and model outputs outside the repository and review staged files before every commit.

## Report a concern privately

Do not open a public issue containing exploit details, credentials, or private data. Use GitHub's **Security → Advisories → Report a vulnerability** flow for this repository when available. If the private reporting option is unavailable, contact the repository owner privately through GitHub. No security contact email is designated by this project.
