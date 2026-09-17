# Aurora Systems GmbH — Internal Product FAQ (Aurora Insights Platform)

Aurora Insights ingests customer usage events from the production API and
surfaces them as dashboards and weekly digest emails to customer success
managers. Data is refreshed hourly; there is no real-time streaming mode as
of the current release.

Access to a customer's dashboard requires that customer to be assigned to you
in the CRM. Dashboards cannot be shared outside the company without going
through the Data Sharing Agreement process, coordinated by Legal.

Known limitation: usage events older than 13 months are archived to cold
storage and are not queryable from the dashboard UI. Requests for archived
data go through a manual retrieval ticket with a 2-business-day SLA.
