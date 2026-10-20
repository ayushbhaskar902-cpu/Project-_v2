import { createElement as h } from 'react'

export interface EvaluationJobStatusItem {
  jobId: number
  status: string
  backfillStatus: string
  backfillErrorClass: string | null
  lastError: string | null
  nextRetryAt: string | null
}

export async function fetchEvaluationJobs(
  sessionId: number,
  fetcher: typeof fetch = fetch,
): Promise<Response> {
  return fetcher(`/api/research/sessions/${sessionId}/evaluation-jobs`, {
    method: 'GET',
    headers: { 'sec-fetch-site': 'same-origin' },
  })
}

function statusLabel(job: EvaluationJobStatusItem): string {
  if (job.status === 'EVALUATED') return 'Evaluation complete'
  if (job.status === 'EVALUABLE') return 'Verified results ready for evaluation'
  if (job.status === 'FAILED' || job.backfillStatus === 'TERMINAL_FAILURE')
    return 'Terminal failure'
  if (job.backfillStatus === 'REVIEW_REQUIRED') return 'Human review required'
  if (job.backfillStatus === 'RETRY_SCHEDULED') {
    return `Retry scheduled${job.nextRetryAt ? ` at ${new Date(job.nextRetryAt).toLocaleTimeString()}` : ''}`
  }
  if (job.backfillStatus === 'IN_PROGRESS') return 'Backfill in progress'
  if (job.backfillErrorClass === 'INCOMPLETE_ARTIFACTS') {
    return 'Artifacts incomplete; waiting for complete results'
  }
  if (job.backfillStatus === 'COMPLETE') return 'Artifacts verified; evaluation pending'
  return 'Waiting for backfill'
}

export function EvaluationJobStatus({
  jobs,
  onRefresh,
  refreshing = false,
}: {
  jobs: EvaluationJobStatusItem[]
  onRefresh: () => void
  refreshing?: boolean
}) {
  if (jobs.length === 0) return null

  const activeCount = jobs.filter((job) => job.status === 'EVALUABLE').length
  const evaluatedCount = jobs.filter((job) => job.status === 'EVALUATED').length

  return h(
    'section',
    {
      'aria-label': 'Evaluation job status',
      'aria-live': 'polite',
      className:
        'mb-4 flex items-center justify-between rounded-lg border border-status-running/40 bg-status-running/10 px-4 py-3 text-caption text-ink',
    },
    h(
      'div',
      null,
      h(
        'p',
        { className: 'font-semibold text-ink' },
        `${activeCount} verified and ready for evaluation; ${evaluatedCount} evaluated.`,
      ),
      h(
        'p',
        { className: 'text-ink-subtle' },
        'Status refresh is read-only and runs every 10 seconds.',
      ),
      h(
        'ul',
        { className: 'mt-2 space-y-1' },
        ...jobs.map((job) => {
          const failed = job.status === 'FAILED' || job.backfillStatus === 'TERMINAL_FAILURE'
          const ready = job.status === 'EVALUABLE' || job.status === 'EVALUATED'
          return h(
            'li',
            {
              key: job.jobId,
              'data-job-status': job.status,
              className: `text-[11px] ${failed ? 'text-pnl-negative-text' : ready ? 'text-pnl-positive-text' : 'text-ink-subtle'}`,
            },
            `Job #${job.jobId}: ${statusLabel(job)}${job.lastError ? ` — ${job.lastError}` : ''}`,
          )
        }),
      ),
    ),
    h(
      'button',
      {
        type: 'button',
        disabled: refreshing,
        onClick: onRefresh,
        className:
          'inline-flex h-7 shrink-0 items-center justify-center gap-1.5 rounded-sm border border-hairline bg-surface-2 px-2 text-body-compact font-medium text-ink transition-colors hover:border-hairline-strong hover:bg-surface-3 active:bg-surface-4 disabled:pointer-events-none disabled:bg-(--btn-disabled-bg) disabled:text-(--btn-disabled-text)',
      },
      'Refresh Status',
    ),
  )
}
