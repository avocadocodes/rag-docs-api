"""
Evaluation dataset for the RAG retrieval system.

A corpus about a fictional API product called "Vela" - a fictional
workflow automation platform.  Each document is a short reference article.
The questions are paired with the IDs of the document(s) that contain the
answer (relevant_doc_ids), used to compute Recall@k and MRR.

Note: relevant_doc_ids here are 1-based indices into DOCUMENTS (not DB primary
keys).  The management command maps them to actual DB IDs after ingestion.

Topic clusters and near-distractor design:
  A. Tokens      - user API tokens (1), service-account/CI tokens (2),
                   webhook signing secrets (3), OAuth client credentials (4)
  B. Runs        - local vela run (5), scheduled/cron runs (6),
                   run retries and backoff (7), run concurrency limits (8),
                   run caching (9)
  C. Errors      - rate limits (10), usage quotas (11), billing tiers (12),
                   timeout settings (13)
  D. Config      - vela.yml syntax (14), environment variables (15),
                   secrets management (16), matrix builds (17)
  E. Integrations- Slack notifications (18), GitHub triggers (19),
                   artifact storage (20), deployment targets (21)
  F. Supporting  - quickstart (22), parallel execution (23),
                   workspace admin (24), audit logs (25),
                   self-hosted runners (26), CLI reference (27),
                   REST API reference (28), RBAC / permissions (29),
                   dependency caching invalidation (30)
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class EvalDocument:
    id: int           # 1-based index, used in questions below
    title: str
    text: str


@dataclass
class EvalQuestion:
    question: str
    relevant_doc_ids: list[int]   # 1-based EvalDocument.id values


DOCUMENTS: list[EvalDocument] = [

    # ── Cluster A: Tokens ────────────────────────────────────────────────────

    EvalDocument(
        id=1,
        title="Vela User API Tokens",
        text=(
            "User API tokens authenticate individual users against the Vela API. "
            "They are generated under Settings > API Tokens in the Vela dashboard "
            "and are scoped to a single workspace. "
            "A new user token inherits the permissions of the user who created it. "
            "By default, user tokens expire after 90 days; this can be extended to "
            "365 days in workspace security settings, but cannot be set to non-expiring. "
            "Tokens are displayed only once at creation time - store them immediately "
            "in a password manager or secrets vault. "
            "To rotate a token, delete it and issue a new one; existing token values "
            "cannot be updated or refreshed in-place."
        ),
    ),

    EvalDocument(
        id=2,
        title="Vela Service-Account and CI Tokens",
        text=(
            "Service-account tokens are intended for unattended processes such as CI "
            "pipelines and deployment scripts. Unlike user tokens, service-account tokens "
            "do not expire automatically - they remain valid until explicitly revoked. "
            "Create a service account under Settings > Service Accounts, then generate "
            "a token for that account. "
            "Service-account tokens carry only the permissions granted to the service "
            "account, keeping the blast radius small if a token is leaked. "
            "Rotate service-account tokens by revoking the current token and issuing a "
            "replacement; both can coexist briefly to allow a zero-downtime rotation. "
            "Never use a personal user token in a CI pipeline - if the user leaves the "
            "organisation, the token is revoked and the pipeline breaks."
        ),
    ),

    EvalDocument(
        id=3,
        title="Vela Webhook Signing Secrets",
        text=(
            "When Vela delivers a webhook payload to your server, it signs the request "
            "body using HMAC-SHA256. "
            "The signing secret is a random 32-byte value you configure under the "
            "'trigger.webhook.secret' key in vela.yml. "
            "Your server must verify the signature by computing HMAC-SHA256 of the raw "
            "request body with the shared secret and comparing it to the value in the "
            "X-Vela-Signature header. "
            "Webhook signing secrets do not expire, but should be rotated annually or "
            "after any suspected compromise. "
            "Rotate a webhook secret by updating the value in vela.yml and simultaneously "
            "updating the secret in your receiving server; a short overlap window is "
            "unavoidable during rotation. "
            "Signing secrets are stored as Vela Secrets, not as user API tokens, and "
            "never appear in run logs."
        ),
    ),

    EvalDocument(
        id=4,
        title="Vela OAuth Client Credentials",
        text=(
            "Vela supports OAuth 2.0 client-credentials flow for machine-to-machine "
            "integrations that need short-lived access tokens. "
            "Register an OAuth application under Settings > OAuth Apps to obtain a "
            "client_id and client_secret. "
            "Exchange these for an access token by posting to "
            "https://auth.vela.io/oauth/token with grant_type=client_credentials. "
            "OAuth access tokens expire after 1 hour and must be refreshed using the "
            "same client credentials; there is no refresh token in the client-credentials "
            "flow. "
            "OAuth tokens are scoped at registration time via a space-separated 'scope' "
            "field (e.g. workflows:read runs:write). "
            "OAuth client credentials are separate from user API tokens and "
            "service-account tokens - they are not visible in Settings > API Tokens."
        ),
    ),

    # ── Cluster B: Runs ───────────────────────────────────────────────────────

    EvalDocument(
        id=5,
        title="Running Workflows Locally with vela run",
        text=(
            "The vela run command executes a workflow on your local machine without "
            "sending anything to the Vela cloud. "
            "Install the CLI with pip install vela-cli, then run vela run from the "
            "directory containing vela.yml. "
            "By default, vela run executes all steps sequentially. "
            "Use --step <name> to run a single named step in isolation. "
            "Use --env KEY=VALUE to inject ephemeral environment variables that are not "
            "committed to vela.yml; these override any matching keys defined in the file. "
            "Local runs do not consume cloud compute minutes and are not visible in the "
            "dashboard run history. "
            "Secrets defined in vela.yml as ${secrets.NAME} are not resolved during local "
            "runs unless you pass --secrets-file pointing to a local .env file."
        ),
    ),

    EvalDocument(
        id=6,
        title="Vela Scheduled (Cron) Runs",
        text=(
            "Scheduled runs execute a workflow automatically on a time-based cadence. "
            "Define the schedule under 'trigger.cron' in vela.yml using standard "
            "five-field cron syntax: minute, hour, day-of-month, month, day-of-week. "
            "All cron times are interpreted in UTC. "
            "Example: '0 8 * * 1-5' runs at 08:00 UTC on every weekday. "
            "If a scheduled run is still executing when the next schedule fires, the "
            "new run is queued and starts immediately after the running one finishes "
            "(runs do not overlap by default). "
            "To allow overlapping scheduled runs, set 'trigger.cron.concurrency: allow'. "
            "Scheduled runs can be paused individually from the dashboard or via "
            "PATCH /workflows/{id} with {'schedule_enabled': false}."
        ),
    ),

    EvalDocument(
        id=7,
        title="Vela Run Retries and Backoff",
        text=(
            "Vela can automatically retry a failed run up to a configurable maximum. "
            "Set 'run.retries' in vela.yml to the maximum number of retry attempts "
            "(default: 0, maximum: 5). "
            "By default, retries use exponential backoff starting at 30 seconds, "
            "doubling each attempt: 30 s, 60 s, 120 s, 240 s, 480 s. "
            "To use a fixed delay instead, set 'run.retry_delay_seconds' to a positive "
            "integer; this disables the exponential backoff. "
            "A run counts as failed for retry purposes if any non-skipped step exits "
            "with a non-zero code. "
            "Each retry is recorded as a separate attempt in the run history and "
            "consumes compute minutes independently. "
            "Retries do not re-download cached layers unless the cache key has changed."
        ),
    ),

    EvalDocument(
        id=8,
        title="Vela Run Concurrency Limits",
        text=(
            "Concurrency limits control how many runs of a workflow can execute "
            "simultaneously across the workspace. "
            "Set 'run.concurrency' in vela.yml to an integer (default: 10). "
            "When the limit is reached, new trigger events queue the run rather than "
            "starting it immediately; queued runs start as slots free up. "
            "Set 'run.concurrency: 1' to enforce that only one run of this workflow "
            "runs at a time - useful for workflows that modify shared state. "
            "Per-workflow concurrency limits are independent of the workspace-level "
            "concurrent-run quota set in your billing plan. "
            "The free tier workspace quota is 3 concurrent runs; Pro is 20; Enterprise "
            "is configurable. "
            "Queued runs time out after 30 minutes by default if they have not yet "
            "started; this is configured via 'run.queue_timeout_minutes'."
        ),
    ),

    EvalDocument(
        id=9,
        title="Vela Run Caching",
        text=(
            "Vela's cache action stores and restores directories between runs to "
            "reduce build time. "
            "Configure it in a workflow step using the vela/cache action with a 'key' "
            "field, typically derived from a hash of a dependency manifest, for example: "
            "key: 'npm-{{ hashFiles(\"package-lock.json\") }}'. "
            "If the exact cache key is not found, Vela falls back to the most recent "
            "cache entry whose key shares the same prefix (e.g. 'npm-'). "
            "To save a new cache at the end of a step, set 'save: true'; to only restore "
            "without saving, set 'save: false'. "
            "Cache entries expire after 7 days of not being accessed, or sooner if the "
            "workspace storage limit is exhausted. "
            "A single cache entry is limited to 5 GB; attempting to save more raises "
            "a CacheSizeExceeded error and the step continues without saving."
        ),
    ),

    # ── Cluster C: Errors / Limits ────────────────────────────────────────────

    EvalDocument(
        id=10,
        title="Vela API Rate Limits",
        text=(
            "The Vela REST API enforces rate limits to ensure fair usage. "
            "Authenticated requests are limited to 1 000 requests per hour per token. "
            "Unauthenticated requests (not recommended) are limited to 60 per hour per "
            "IP address. "
            "When a limit is exceeded, the API returns HTTP 429 Too Many Requests with a "
            "Retry-After header indicating the number of seconds to wait. "
            "The rate limit window resets on a rolling-hourly basis, not at the top of "
            "the clock hour. "
            "The remaining quota for the current window is reported in the "
            "X-RateLimit-Remaining response header on every successful request. "
            "Webhook delivery calls originating from Vela do not count against your "
            "API rate limit."
        ),
    ),

    EvalDocument(
        id=11,
        title="Vela Usage Quotas",
        text=(
            "Usage quotas cap total resource consumption within a billing period "
            "(calendar month). "
            "The free tier includes 2 000 compute-minutes and 1 GB of cache storage "
            "per month. "
            "The Pro tier includes 20 000 compute-minutes and 20 GB of cache storage. "
            "Enterprise tiers negotiate custom quotas. "
            "When the compute-minute quota is exhausted, all new run triggers are "
            "rejected with HTTP 402 Payment Required until the next billing cycle or "
            "until you upgrade your plan. "
            "Quota consumption is visible in real time on the Settings > Usage page. "
            "Compute minutes are counted from the moment a run starts executing to the "
            "moment it finishes or is cancelled; queued time is not counted."
        ),
    ),

    EvalDocument(
        id=12,
        title="Vela Billing Tiers",
        text=(
            "Vela offers three billing tiers: Free, Pro, and Enterprise. "
            "The Free tier is available without a credit card and supports one workspace, "
            "up to 3 team members, 2 000 compute-minutes/month, and 1 GB cache. "
            "Pro is $49 per workspace per month and supports unlimited team members, "
            "20 000 compute-minutes/month, 20 GB cache, and priority support. "
            "Enterprise pricing is negotiated and includes custom compute-minute "
            "allowances, SSO/SAML, audit logging, and a dedicated SLA. "
            "Compute-minutes used beyond the included quota are billed at $0.004 per "
            "minute on Pro; overage is not available on Free. "
            "Billing is managed under Settings > Billing; invoices are issued on the "
            "first day of each month and emailed to the workspace owner."
        ),
    ),

    EvalDocument(
        id=13,
        title="Vela Step and Run Timeout Settings",
        text=(
            "Vela enforces two levels of timeout to prevent runaway workloads. "
            "Step-level timeout: set 'timeout_minutes' on any step; if the step does "
            "not complete within that duration, it is killed and marked as timed-out "
            "(default: 60 minutes, maximum: 360 minutes). "
            "Run-level timeout: set 'run.timeout_minutes' in vela.yml; if the entire "
            "run exceeds this duration it is terminated and all running steps are "
            "killed (default: 120 minutes, maximum: 720 minutes). "
            "A timed-out run is treated as a failure and triggers retries if "
            "'run.retries' is configured. "
            "Queued runs that have not yet started are governed by "
            "'run.queue_timeout_minutes', not by the run-level timeout."
        ),
    ),

    # ── Cluster D: Config ─────────────────────────────────────────────────────

    EvalDocument(
        id=14,
        title="vela.yml Syntax Reference",
        text=(
            "vela.yml is the single configuration file that defines a Vela workflow. "
            "It must be placed at the root of your repository. "
            "Top-level keys are: 'name' (string, required), 'trigger' (object), "
            "'env' (map of string to string), 'run' (object), and 'steps' (list). "
            "Each step requires at minimum an 'image' (Docker image) and a 'run' "
            "command or 'uses' action reference. "
            "Step names must be unique within a workflow and may only contain "
            "alphanumeric characters, hyphens, and underscores. "
            "YAML anchors and aliases are supported, allowing you to define reusable "
            "blocks at the top of the file and reference them with the '*alias' syntax. "
            "The maximum file size for vela.yml is 1 MB; workflows exceeding this limit "
            "are rejected at parse time with error code CONFIG_TOO_LARGE."
        ),
    ),

    EvalDocument(
        id=15,
        title="Vela Environment Variables",
        text=(
            "Environment variables in Vela can be set at three scopes: workspace, "
            "workflow, and step. "
            "Workspace-level variables are defined in Settings > Environment and are "
            "automatically injected into every run in that workspace. "
            "Workflow-level variables are defined under the top-level 'env' key in "
            "vela.yml and override workspace variables of the same name. "
            "Step-level variables are defined under 'env' within a step block and "
            "override both workspace and workflow variables. "
            "Vela also injects a set of built-in variables at runtime, including "
            "VELA_RUN_ID, VELA_WORKFLOW_NAME, VELA_COMMIT_SHA, and VELA_BRANCH. "
            "Environment variable names are case-sensitive. "
            "To reference a Vela Secret as an environment variable, use the "
            "'secrets' block inside a step rather than the top-level 'env' map."
        ),
    ),

    EvalDocument(
        id=16,
        title="Vela Secrets Management",
        text=(
            "Vela Secrets stores sensitive values outside vela.yml so they are never "
            "committed to source control. "
            "Secrets are encrypted at rest using AES-256-GCM and are decrypted only "
            "when injected into a running step. "
            "Create secrets via the dashboard (Settings > Secrets), the CLI "
            "(vela secret set NAME VALUE), or the REST API (POST /secrets). "
            "Reference a secret inside a step by listing it in the step's 'secrets' "
            "block; the secret is injected as an environment variable whose name "
            "matches the secret name. "
            "Secrets are scoped to either the workspace or a specific workflow; "
            "workflow-scoped secrets take precedence over workspace-scoped secrets "
            "with the same name. "
            "Secret values are redacted from run logs - any output that matches a "
            "secret value is replaced with '***'."
        ),
    ),

    EvalDocument(
        id=17,
        title="Vela Matrix Builds",
        text=(
            "Matrix builds run the same workflow steps across multiple combinations "
            "of variables, useful for cross-platform or multi-version testing. "
            "Define a matrix under 'run.matrix' as a map of variable names to lists "
            "of values. "
            "Vela generates one run per unique combination; for example, a matrix with "
            "two OS values and three Node.js versions produces six runs. "
            "Matrix variables are available as environment variables inside steps using "
            "the VELA_MATRIX_ prefix (e.g. VELA_MATRIX_NODE_VERSION). "
            "Use 'run.matrix.exclude' to omit specific combinations. "
            "By default all matrix runs execute concurrently, up to the workflow's "
            "'run.concurrency' limit. "
            "Set 'run.matrix.fail_fast: true' to cancel all remaining matrix runs "
            "as soon as any single run fails."
        ),
    ),

    # ── Cluster E: Integrations ───────────────────────────────────────────────

    EvalDocument(
        id=18,
        title="Vela Slack Notifications",
        text=(
            "Vela can post run status messages to a Slack channel using the "
            "vela/notify action with 'provider: slack'. "
            "Configure the integration by storing your Slack incoming-webhook URL as "
            "a Vela Secret and referencing it with 'webhook_url: ${secrets.SLACK_WEBHOOK}'. "
            "Notifications support conditional delivery: set 'on: failure', 'on: success', "
            "or 'on: always' (default). "
            "The message body is a Jinja2 template; available variables include "
            "{{ run.status }}, {{ run.duration }}, {{ run.url }}, {{ workflow.name }}, "
            "and {{ commit.sha }}. "
            "To post to multiple channels, add multiple vela/notify steps with different "
            "webhook URLs. "
            "Slack notifications are delivered asynchronously after the run completes "
            "and do not affect run status or duration."
        ),
    ),

    EvalDocument(
        id=19,
        title="Vela GitHub Integration and Triggers",
        text=(
            "Connecting Vela to GitHub enables push- and pull-request-triggered runs. "
            "Install the Vela GitHub App on your repository from Settings > Integrations. "
            "Once installed, push events to any branch trigger a run by default; "
            "filter to specific branches with 'trigger.push.branches' in vela.yml. "
            "Pull-request events (opened, synchronised, reopened) trigger runs on the "
            "head commit of the PR branch. "
            "Vela posts a commit status back to GitHub with the run result, visible in "
            "the PR checks UI. "
            "To trigger only on specific file paths, use 'trigger.push.paths' with "
            "glob patterns. "
            "For monorepos, set 'trigger.push.paths_ignore' to suppress runs when "
            "only unrelated files changed."
        ),
    ),

    EvalDocument(
        id=20,
        title="Vela Artifact Storage",
        text=(
            "The vela/artifacts action uploads files from a run to Vela's artifact "
            "store for later download or use by downstream workflows. "
            "Specify a 'path' glob to select files (e.g. 'dist/**') and an optional "
            "'name' to label the artifact set. "
            "Artifacts are retained for 30 days by default; Pro workspaces can "
            "configure retention up to 180 days under Settings > Artifacts. "
            "Download artifacts from the dashboard run detail page or via the API: "
            "GET /runs/{run_id}/artifacts returns a list, and "
            "GET /artifacts/{artifact_id}/download returns the file stream. "
            "Maximum artifact size per run is 10 GB. "
            "Artifacts are distinct from the run cache - artifacts persist across "
            "workspace storage-limit resets and are not subject to the 7-day cache "
            "expiry."
        ),
    ),

    EvalDocument(
        id=21,
        title="Vela Deployment Targets",
        text=(
            "Vela deployment targets represent named environments (e.g. staging, "
            "production) to which a workflow can push a release. "
            "Define targets under Settings > Deployments; each target has a name, "
            "an optional approval gate, and a set of environment variables injected "
            "only during deployments to that target. "
            "Trigger a deployment run from the CLI with "
            "vela deploy --target production --ref v1.2.3. "
            "If the target has an approval gate, the run pauses after the build phase "
            "and waits for a workspace admin to approve via the dashboard or "
            "POST /deployments/{id}/approve. "
            "Deployment history and approval audit trails are available under the "
            "Deployments tab in the workspace."
        ),
    ),

    # ── Cluster F: Supporting docs ────────────────────────────────────────────

    EvalDocument(
        id=22,
        title="Vela Quickstart Guide",
        text=(
            "Vela is a workflow automation platform for engineering teams. "
            "Install the CLI: pip install vela-cli. "
            "Authenticate: vela login --token <YOUR_TOKEN>. "
            "Create a vela.yml at the repository root defining at least one step with "
            "an 'image' and a 'run' command. "
            "Execute the workflow locally: vela run. "
            "Push the repository to trigger cloud runs if the GitHub integration is "
            "configured. "
            "View run output in the Vela dashboard under Workflows > Run History. "
            "The quickstart template at https://docs.vela.io/quickstart/template "
            "provides a working vela.yml with lint, test, and deploy steps."
        ),
    ),

    EvalDocument(
        id=23,
        title="Vela Parallel Step Execution",
        text=(
            "By default, Vela steps execute sequentially in the order they are listed. "
            "Assign steps to a named 'group' to run them concurrently: all steps "
            "sharing the same group name start simultaneously. "
            "Use 'depends_on' to declare explicit dependencies between steps, forming "
            "a directed acyclic graph (DAG); a step only starts once all listed "
            "dependencies have completed successfully. "
            "'group' and 'depends_on' can be combined: steps in the same group that "
            "also list 'depends_on' wait for their dependencies before joining the "
            "group's concurrent batch. "
            "If a dependency fails, downstream steps are skipped unless the dependent "
            "step sets 'continue_on_error: true'. "
            "Group names have no significance beyond scoping concurrency - they need "
            "not be unique across the file."
        ),
    ),

    EvalDocument(
        id=24,
        title="Vela Workspace Administration",
        text=(
            "A workspace is the top-level organisational unit in Vela. "
            "Each workspace has isolated secrets, tokens, storage, and billing. "
            "Workspace administrators can invite team members under Settings > Members "
            "and assign them one of three roles: Viewer, Developer, or Admin. "
            "Viewers can see runs and logs but cannot trigger or edit workflows. "
            "Developers can trigger, edit, and cancel workflows and manage secrets "
            "scoped to workflows they own. "
            "Admins have full access including billing and workspace settings. "
            "A workspace can be renamed or deleted by an Admin; deletion permanently "
            "removes all workflows, secrets, tokens, and run history."
        ),
    ),

    EvalDocument(
        id=25,
        title="Vela Audit Logs",
        text=(
            "Vela records an audit log entry for every security-relevant action in a "
            "workspace. "
            "Audited events include: token creation and revocation, secret creation, "
            "update, and deletion, member invitation and removal, role changes, "
            "workflow creation and deletion, and deployment approvals. "
            "Audit logs are retained for 90 days on Free and Pro tiers. "
            "Enterprise workspaces can configure export to an S3 bucket or Splunk "
            "endpoint for long-term retention. "
            "Access audit logs via Settings > Audit Log in the dashboard or via "
            "GET /audit-logs (paginated, max 100 entries per page). "
            "Audit log entries include the actor's user ID, IP address, timestamp, "
            "event type, and a JSON payload describing what changed."
        ),
    ),

    EvalDocument(
        id=26,
        title="Vela Self-Hosted Runners",
        text=(
            "Self-hosted runners let you execute Vela workflow runs on your own "
            "infrastructure instead of Vela's cloud compute. "
            "Install the runner agent with: curl -fsSL https://get.vela.io/runner | sh. "
            "Authenticate the runner by setting the VELA_RUNNER_TOKEN environment "
            "variable to a service-account token generated for the runner. "
            "Tag the runner with one or more labels (e.g. gpu, arm64) and target it "
            "in vela.yml with 'run.runs_on: [gpu]'. "
            "Self-hosted runner runs do not consume cloud compute minutes from your "
            "billing quota. "
            "The runner agent must be able to reach https://api.vela.io and must have "
            "Docker (or the configured container runtime) installed. "
            "Multiple runner agents can be registered to a workspace to increase "
            "parallelism."
        ),
    ),

    EvalDocument(
        id=27,
        title="Vela CLI Reference",
        text=(
            "The Vela CLI (vela) provides subcommands for all common operations. "
            "vela login --token <TOKEN> authenticates the CLI session; credentials are "
            "stored in ~/.vela/credentials. "
            "vela run [--step NAME] [--env K=V] executes the local workflow. "
            "vela run trigger --workflow <id> triggers a remote run via the API. "
            "vela secret set NAME VALUE and vela secret delete NAME manage secrets. "
            "vela deploy --target <name> --ref <git-ref> triggers a deployment run. "
            "vela logs <run-id> streams logs for a remote run. "
            "vela config set workspace <id> switches the active workspace. "
            "Pass --help to any subcommand for full flag documentation. "
            "The CLI respects the VELA_TOKEN and VELA_WORKSPACE environment variables "
            "as alternatives to the stored credentials file."
        ),
    ),

    EvalDocument(
        id=28,
        title="Vela REST API Reference",
        text=(
            "The Vela REST API base URL is https://api.vela.io/v1. "
            "All requests must include Authorization: Bearer <token>. "
            "Key endpoints: GET /workflows (list workflows), "
            "POST /workflows/{id}/run (trigger a run), "
            "GET /runs/{run_id} (run status), "
            "GET /runs/{run_id}/logs (stream logs), "
            "DELETE /runs/{run_id} (cancel a run), "
            "GET /secrets (list secrets), POST /secrets (create secret), "
            "GET /audit-logs (audit events, paginated). "
            "All list endpoints support pagination via 'page' and 'per_page' query "
            "params (default per_page: 25, max: 100). "
            "Dates are returned in ISO 8601 format in UTC. "
            "Errors follow RFC 7807 Problem Details: JSON body with 'type', 'title', "
            "'status', and 'detail' fields."
        ),
    ),

    EvalDocument(
        id=29,
        title="Vela RBAC and Permissions",
        text=(
            "Vela uses role-based access control (RBAC) at the workspace level. "
            "The three built-in roles are Viewer, Developer, and Admin (see Workspace "
            "Administration for role capabilities). "
            "In addition to workspace roles, individual workflows can have fine-grained "
            "access policies: a workflow owner can restrict who may trigger, edit, or "
            "view a specific workflow using the 'permissions' block in vela.yml. "
            "Service accounts are granted roles the same way as human users and their "
            "token permissions are bounded by their assigned role. "
            "Enterprise workspaces can define custom roles with granular permission "
            "sets via Settings > Custom Roles. "
            "Permission changes take effect immediately without requiring a token "
            "reissue - tokens inherit the current role of the user or service account "
            "that owns them."
        ),
    ),

    EvalDocument(
        id=30,
        title="Vela Cache Invalidation and Dependency Updates",
        text=(
            "Cache entries in Vela are invalidated by changing the cache key. "
            "Because cache keys are typically a hash of a dependency manifest "
            "(e.g. package-lock.json), updating a dependency automatically produces "
            "a new key and bypasses the stale cache. "
            "To force a full cache bust without changing dependencies, append a "
            "manual version suffix to the key: e.g. 'npm-v2-{{ hashFiles(...) }}'. "
            "You can also delete a specific cache entry via "
            "DELETE /caches/{cache_key} in the REST API. "
            "Deleting a cache entry does not affect subsequent runs until they "
            "encounter a key miss and rebuild the cache. "
            "Stale cache entries that have not been accessed for 7 days are pruned "
            "automatically; the pruning job runs nightly at 02:00 UTC."
        ),
    ),
]


QUESTIONS: list[EvalQuestion] = [

    # ── Direct factual (keyword overlap with correct doc) ─────────────────────

    EvalQuestion(
        "How long do user API tokens last before they expire?",
        [1],
    ),
    EvalQuestion(
        "Which type of Vela token does not expire automatically and is intended for CI pipelines?",
        [2],
    ),
    EvalQuestion(
        "What HTTP header does Vela include when it signs a webhook delivery?",
        [3],
    ),
    EvalQuestion(
        "How long do OAuth access tokens issued by Vela's client-credentials flow remain valid?",
        [4],
    ),
    EvalQuestion(
        "How do I execute a single named step locally without running the whole workflow?",
        [5],
    ),
    EvalQuestion(
        "What cron expression would schedule a Vela workflow at 08:00 UTC every weekday?",
        [6],
    ),
    EvalQuestion(
        "What is the default exponential backoff delay sequence when Vela retries a failed run?",
        [7],
    ),
    EvalQuestion(
        "What happens to a queued Vela run that has not started within the queue timeout?",
        [8],
    ),
    EvalQuestion(
        "What is the maximum size of a single Vela cache entry?",
        [9],
    ),
    EvalQuestion(
        "What HTTP status code does the Vela API return when the rate limit is exceeded?",
        [10],
    ),
    EvalQuestion(
        "What HTTP status is returned when a workspace exhausts its monthly compute-minute quota?",
        [11],
    ),
    EvalQuestion(
        "How much does a Pro Vela workspace cost per month?",
        [12],
    ),
    EvalQuestion(
        "What is the default step-level timeout in Vela, and what is the maximum allowed?",
        [13],
    ),
    EvalQuestion(
        "What are the required top-level keys in a vela.yml file?",
        [14],
    ),
    EvalQuestion(
        "Which built-in environment variables does Vela inject into every run?",
        [15],
    ),
    EvalQuestion(
        "How do I force all matrix runs to stop as soon as one of them fails?",
        [17],
    ),
    EvalQuestion(
        "What template variables are available in a Vela Slack notification message?",
        [18],
    ),
    EvalQuestion(
        "How do I trigger a Vela run only when specific file paths change on a push?",
        [19],
    ),
    EvalQuestion(
        "How long are artifacts retained by default, and where is that configured for Pro workspaces?",
        [20],
    ),
    EvalQuestion(
        "How do I trigger a deployment to a named environment using the Vela CLI?",
        [21],
    ),

    # ── Paraphrased (avoid doc keywords; favour semantic/dense retrieval) ─────

    EvalQuestion(
        "I want a machine credential for automated scripts that stays valid indefinitely unless manually cancelled.",
        [2],
    ),
    EvalQuestion(
        "What is the procedure for verifying that an incoming event from Vela has not been tampered with?",
        [3],
    ),
    EvalQuestion(
        "How can I test my workflow on my laptop without spending any cloud build minutes?",
        [5],
    ),
    EvalQuestion(
        "My workflow mutates a shared database and must never run more than once at a time. How do I enforce that?",
        [8],
    ),
    EvalQuestion(
        "I need my build to reuse previously downloaded packages across separate executions to keep things fast.",
        [9],
    ),
    EvalQuestion(
        "Our team is hitting an error that blocks new jobs from starting because we have used up all our allocated processing time this month.",
        [11],
    ),
    EvalQuestion(
        "Sensitive credentials must not show up in the logs even if a step accidentally prints them.",
        [16],
    ),
    EvalQuestion(
        "I want to verify the same pipeline against three versions of Python and two operating systems simultaneously.",
        [17],
    ),
    EvalQuestion(
        "How do I set up Vela to notify my team's chat channel when a job finishes with an error?",
        [18],
    ),
    EvalQuestion(
        "Where can I find a record of who created or revoked credentials in my workspace?",
        [25],
    ),

    # ── Disambiguation (keywords appear in multiple docs; only one answers) ───

    EvalQuestion(
        "How do I rotate a token to minimise downtime, allowing the old and new token to coexist briefly?",
        [2],
        # Doc 1 mentions rotation but says you must delete first; Doc 2 explicitly
        # describes zero-downtime coexistence rotation for service-account tokens.
    ),
    EvalQuestion(
        "Where in vela.yml do I configure the secret used to authenticate inbound webhook payloads?",
        [3],
        # Doc 5/16 both discuss 'secrets' in vela.yml, but only Doc 3 covers the
        # trigger.webhook.secret key for validating inbound requests.
    ),
    EvalQuestion(
        "What happens when a scheduled run is already executing and its next scheduled time arrives - does it start a second copy?",
        [6],
        # Doc 8 covers concurrency limits; Doc 7 covers retries; only Doc 6 covers
        # the overlap behaviour specific to cron-scheduled runs.
    ),
    EvalQuestion(
        "A step that times out - does Vela treat it as a failure for the purposes of triggering automatic retries?",
        [13],
        # Doc 7 covers retries and mentions 'non-zero exit'; only Doc 13 explicitly
        # states that timed-out runs are treated as failures and trigger retries.
    ),
    EvalQuestion(
        "What is the difference between a cache entry expiring and an artifact expiring in Vela?",
        [9, 20],
        # Both docs are needed to answer the comparison question; wrong to label
        # only one.
    ),
    EvalQuestion(
        "How do I prevent a Vela workflow from being triggered on pushes to irrelevant paths in a monorepo?",
        [19],
        # Doc 14 discusses vela.yml syntax; Doc 15 mentions VELA_BRANCH; only
        # Doc 19 covers trigger.push.paths_ignore for monorepo path filtering.
    ),
    EvalQuestion(
        "I need compute to run on my own GPU servers instead of Vela's cloud. How do I configure that?",
        [26],
        # Doc 8 mentions concurrency; Doc 12 mentions compute minutes; only Doc 26
        # covers self-hosted runners and the runs_on label mechanism.
    ),
    EvalQuestion(
        "A developer left the company and their token was revoked. Will this break any CI pipelines that used their credentials?",
        [2],
        # Doc 1 and Doc 29 discuss tokens and permissions; only Doc 2 explicitly
        # warns about this exact scenario and recommends service-account tokens.
    ),
    EvalQuestion(
        "How do I forcibly invalidate a cached build without bumping any dependency version?",
        [30],
        # Doc 9 explains cache keys; only Doc 30 explains the manual version-suffix
        # bust technique and the DELETE /caches endpoint.
    ),
    EvalQuestion(
        "What permission level does someone need to approve a deployment in Vela?",
        [21, 24],
        # Doc 21 says a workspace admin must approve; Doc 24 defines what Admin role
        # means; both together fully answer the question.
    ),
]


OUT_OF_CORPUS_QUESTIONS: list[EvalQuestion] = [
    EvalQuestion(
        question="How do I configure Vela to push container images to Docker Hub?",
        relevant_doc_ids=[],
    ),
    EvalQuestion(
        question="What is the maximum file size for a Vela artifact when using S3-compatible storage backends?",
        relevant_doc_ids=[],
    ),
    EvalQuestion(
        question="How do I configure SSO with Okta in a Vela free-tier workspace?",
        relevant_doc_ids=[],
    ),
    EvalQuestion(
        question="Can I use Vela to schedule database migrations automatically on every deploy?",
        relevant_doc_ids=[],
    ),
    EvalQuestion(
        question="What is the Vela API endpoint for creating a new workspace programmatically?",
        relevant_doc_ids=[],
    ),
]
