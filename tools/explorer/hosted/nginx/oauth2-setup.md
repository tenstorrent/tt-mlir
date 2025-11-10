# OAuth2 Authentication Setup for TT-Explorer

For a more production-ready deployment, TT-Explorer uses [oauth2-proxy](https://oauth2-proxy.github.io/oauth2-proxy/) for authentication. This guide covers setup for Microsoft Entra ID (aka Azure AD) and generic OIDC providers.

## Azure Entra ID (formerly Azure AD) Setup

### 1. Register Application in Azure Portal

1. Navigate to [Azure Portal](https://portal.azure.com) → Azure Active Directory → App registrations
2. Click "New registration"
3. Configure:
   - **Name**: TT-Explorer
   - **Supported account types**: Accounts in this organizational directory only
   - **Redirect URI**: Web → `https://your-domain.com/oauth2/callback`
4. Click "Register"

### 2. Configure Application

After registration:

1. **Copy Application (client) ID** → Use as `OAUTH2_CLIENT_ID`
2. **Copy Directory (tenant) ID** → Use as `OAUTH2_AZURE_TENANT`
3. Go to "Certificates & secrets" → "New client secret"
   - Description: TT-Explorer
   - Expires: Choose appropriate expiration
   - **Copy the secret value** → Use as `OAUTH2_CLIENT_SECRET` (save immediately, cannot retrieve later)

### 3. Configure Environment Variables

Copy `env.example` to `.env` and configure:

```bash
# Required for Microsoft Entra ID
OAUTH2_PROVIDER=entra-id
OAUTH2_OIDC_ISSUER_URL=https://login.microsoftonline.com/your-tenant-id-from-azure/v2.0
OAUTH2_CLIENT_ID=your-application-client-id-from-azure
OAUTH2_CLIENT_SECRET=your-client-secret-from-azure

# Generate cookie secret (32 bytes, base64 encoded)
OAUTH2_COOKIE_SECRET=$(python -c 'import os,base64; print(base64.urlsafe_b64encode(os.urandom(32)).decode())')

# Update your domain
FQDN=explorer.yourcompany.com
OAUTH2_REDIRECT_URL=https://explorer.yourcompany.com/oauth2/callback

# Optional: restrict email domains (use * to allow all in your tenant)
OAUTH2_EMAIL_DOMAINS=yourcompany.com
```

### 4. Optional: Restrict Access by Group

To restrict access to specific Azure AD groups:

1. In Azure Portal → App registrations → Your app → Token configuration
2. Add optional claim for "groups"
3. Add to `.env`:
   ```bash
   OAUTH2_ALLOWED_GROUPS=group-id-1,group-id-2
   ```

## Generic OIDC Provider Setup

For non-Azure OIDC providers (Okta, Auth0, Keycloak, etc.):

### 1. Register Application

Register an OAuth2/OIDC application with your provider:
- **Redirect URI**: `https://your-domain.com/oauth2/callback`
- **Grant type**: Authorization Code
- **Scopes**: openid, profile, email

### 2. Configure Environment Variables

```bash
OAUTH2_PROVIDER=oidc
OAUTH2_OIDC_ISSUER_URL=https://your-oidc-provider.com
OAUTH2_CLIENT_ID=your-client-id
OAUTH2_CLIENT_SECRET=your-client-secret
OAUTH2_COOKIE_SECRET=$(python -c 'import os,base64; print(base64.urlsafe_b64encode(os.urandom(32)).decode())')
FQDN=explorer.yourcompany.com
OAUTH2_REDIRECT_URL=https://explorer.yourcompany.com/oauth2/callback
```

## Provider-Specific Examples

### Google

```bash
OAUTH2_PROVIDER=google
OAUTH2_CLIENT_ID=your-client-id.apps.googleusercontent.com
OAUTH2_CLIENT_SECRET=your-client-secret
OAUTH2_EMAIL_DOMAINS=yourcompany.com
```

### GitHub

```bash
OAUTH2_PROVIDER=github
OAUTH2_CLIENT_ID=your-github-oauth-app-client-id
OAUTH2_CLIENT_SECRET=your-github-oauth-app-client-secret
OAUTH2_GITHUB_ORG=your-github-org  # Optional: restrict to org
OAUTH2_GITHUB_TEAM=your-github-team  # Optional: restrict to team
```

## Testing

1. Start services: `make up`
2. Navigate to `https://your-domain.com`
3. You should be redirected to your OAuth2 provider's login page
4. After successful authentication, you'll be redirected back to TT-Explorer

## Troubleshooting

### Redirect URI Mismatch
- Ensure `OAUTH2_REDIRECT_URL` matches exactly what's registered in your OAuth2 provider
- Include the protocol (https://) and path (/oauth2/callback)

### Cookie Errors
- Ensure `OAUTH2_COOKIE_SECRET` is exactly 32 bytes when base64 decoded
- Generate new secret: `python -c 'import os,base64; print(base64.urlsafe_b64encode(os.urandom(32)).decode())'`

### HTTPS Required
- OAuth2 cookies require HTTPS in production (`OAUTH2_COOKIE_SECURE=true`)
- For local testing only, set `OAUTH2_COOKIE_SECURE=false`

### Check oauth2-proxy logs
```bash
docker logs tt-explorer-oauth2-proxy
```

## Additional Resources

- [oauth2-proxy documentation](https://oauth2-proxy.github.io/oauth2-proxy/)
- [Azure AD configuration](https://oauth2-proxy.github.io/oauth2-proxy/docs/configuration/oauth_provider#azure-auth-provider)
- [Nginx auth_request integration](https://oauth2-proxy.github.io/oauth2-proxy/configuration/integration/)

